import { OpenAI } from "openai";
import * as fs from "fs";
import * as path from "path";
import { isEmpty } from "lodash";
import moment from "moment";
import dotenv from "dotenv";
import {
  shouldResetChatHistory,
  systemPrompt,
  updateLastMessageTime,
} from "../../config/llm-config";
import { FunctionCall, Message } from "../../type";
import { combineFunction } from "../../utils";
import { llmFuncMap, llmTools } from "../../config/llm-tools";
import { ChatWithLLMStreamFunction } from "../interface";
import { chatHistoryDir } from "../../utils/dir";

dotenv.config();

const lmStudioBaseURL =
  process.env.LMSTUDIO_BASE_URL || "http://localhost:1234/v1";
const lmStudioAPIKey = process.env.LMSTUDIO_API_KEY || "lm-studio";
const lmStudioModel = process.env.LMSTUDIO_MODEL || "local-model";
const enableTools = process.env.LMSTUDIO_ENABLE_TOOLS === "true";

const openai = new OpenAI({
  baseURL: lmStudioBaseURL,
  apiKey: lmStudioAPIKey,
});

// Use whichever model LM Studio currently has loaded, so the chatbot never
// forces a model swap (e.g. while a heavy model is loaded for another job).
// LMSTUDIO_MODEL is preferred when several are loaded, and is the fallback
// (loaded on demand by LM Studio) when none is.
const lmStudioApiV0 = lmStudioBaseURL.replace(/\/v1\/?$/, "/api/v0");
let lastUsedModel = "";

const resolveModel = async (): Promise<string> => {
  let model = lmStudioModel;
  try {
    const res = await fetch(`${lmStudioApiV0}/models`, {
      signal: AbortSignal.timeout(3000),
    });
    const { data } = (await res.json()) as {
      data: { id: string; type: string; state: string }[];
    };
    const loaded = data
      .filter((m) => (m.type === "llm" || m.type === "vlm") && m.state === "loaded")
      .map((m) => m.id);
    if (loaded.length > 0) {
      model = loaded.includes(lmStudioModel) ? lmStudioModel : loaded[0];
    }
  } catch (error) {
    console.error("Could not list LM Studio models, using", lmStudioModel, error);
  }
  if (model !== lastUsedModel) {
    console.log(`LM Studio model: ${model}`);
    lastUsedModel = model;
  }
  return model;
};

const chatHistoryFileName = `lmstudio_chat_history_${moment().format(
  "YYYY-MM-DD_HH-mm-ss"
)}.json`;

const messages: Message[] = [
  {
    role: "system",
    content: systemPrompt,
  },
];

const resetChatHistory = (): void => {
  messages.length = 0;
  messages.push({
    role: "system",
    content: systemPrompt,
  });
};

// Thinking control for reasoning models (e.g. Qwen3.8): LMSTUDIO_THINKING=
//   auto   - (default) answer without thinking first; the model replies
//            THINK_MARKER when a request needs real reasoning, and is then
//            re-asked with thinking on
//   always - always let the model think
//   never  - never think
// Only reasoning_effort "none" turns thinking off through LM Studio; the
// "/no_think" tag and chat_template_kwargs are ignored by Qwen3.8.
const thinkingMode = (process.env.LMSTUDIO_THINKING || "auto").toLowerCase();
const THINK_MARKER = "[THINK]";
const ESCALATE_RULE =
  `\n\nIf answering well needs careful multi-step reasoning, calculation, ` +
  `planning or writing code, reply with exactly ${THINK_MARKER} and nothing ` +
  `else. Otherwise answer directly.`;

type PassResult = {
  escalate: boolean;
  answer: string;
  functionCallsPackages: any[];
};

// One streamed completion over the shared history.
//   "first"     - no thinking, may escalate (auto mode's first pass)
//   "reasoning" - thinking on (auto mode's second pass)
//   "single"    - the only pass, per LMSTUDIO_THINKING always/never
const streamPass = async (
  kind: "first" | "reasoning" | "single",
  partialCallback: (partial: string) => void,
  partialThinkingCallback?: (partialThinking: string) => void
): Promise<PassResult> => {
  const passMessages =
    kind === "first"
      ? messages.map((m, i) =>
          i === 0 && m.role === "system"
            ? { ...m, content: `${m.content}${ESCALATE_RULE}` }
            : m
        )
      : messages;
  const request: any = {
    model: await resolveModel(),
    messages: passMessages,
    stream: true,
    tools: enableTools ? llmTools : undefined,
  };
  if (kind === "first" || (kind === "single" && thinkingMode === "never")) {
    request.reasoning_effort = "none";
  }
  const stream = (await openai.chat.completions.create(request)) as any;

  let answer = "";
  // on the first pass, hold the text back until it cannot be THINK_MARKER,
  // so the marker is never shown or spoken
  let holding = kind === "first";
  const functionCallsPackages: any[] = [];
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta ?? {};
    // thinking models stream their reasoning separately; show it on the
    // "Thinking" screen (it is not spoken)
    if (delta.reasoning_content) {
      partialThinkingCallback?.(delta.reasoning_content);
    }
    if (delta.content) {
      answer += delta.content;
      if (!holding) {
        partialCallback(delta.content);
      } else {
        const start = answer.trimStart();
        if (start.startsWith(THINK_MARKER)) {
          stream.controller?.abort();
          return { escalate: true, answer: "", functionCallsPackages: [] };
        }
        if (!THINK_MARKER.startsWith(start)) {
          holding = false;
          partialCallback(answer);
        }
      }
    }
    if (delta.tool_calls) {
      functionCallsPackages.push(...delta.tool_calls);
    }
  }
  if (holding && answer) {
    // a very short answer that is still a prefix of the marker
    partialCallback(answer);
  }
  return { escalate: false, answer, functionCallsPackages };
};

const chatWithLLMStream: ChatWithLLMStreamFunction = async (
  inputMessages: Message[] = [],
  partialCallback: (partial: string) => void,
  endCallback: () => void,
  partialThinkingCallback?: (partialThinking: string) => void,
  invokeFunctionCallback?: (functionName: string, result?: string) => void
): Promise<void> => {
  if (shouldResetChatHistory()) {
    resetChatHistory();
  }
  updateLastMessageTime();

  let endResolve: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    endResolve = resolve;
  }).finally(() => {
    fs.writeFileSync(
      path.join(chatHistoryDir, chatHistoryFileName),
      JSON.stringify(messages, null, 2)
    );
  });

  messages.push(...inputMessages);

  try {
    let pass = await streamPass(
      thinkingMode === "auto" ? "first" : "single",
      partialCallback,
      partialThinkingCallback
    );
    if (pass.escalate) {
      console.log("LM Studio: model asked to think, re-asking with reasoning");
      pass = await streamPass("reasoning", partialCallback, partialThinkingCallback);
    }

    const answer = pass.answer;
    const functionCalls = combineFunction(pass.functionCallsPackages);

    messages.push({
      role: "assistant",
      content: answer,
      tool_calls: isEmpty(functionCalls) ? undefined : functionCalls,
    });

    if (!isEmpty(functionCalls)) {
      const results = await Promise.all(
        functionCalls.map(async (call: FunctionCall) => {
          const {
            function: { arguments: argString, name },
            id,
          } = call;
          let args: Record<string, any> = {};
          try {
            args = JSON.parse(argString || "{}");
          } catch {
            console.error(
              `Error parsing arguments for function ${name}:`,
              argString
            );
          }
          const func = llmFuncMap[name! as string];
          invokeFunctionCallback?.(name! as string);

          if (func) {
            return [
              id,
              await func(args)
                .then((res) => {
                  invokeFunctionCallback?.(name! as string, res);
                  return res;
                })
                .catch((err) => {
                  console.error(`Error executing function ${name}:`, err);
                  return `Error executing function ${name}: ${err.message}`;
                }),
            ];
          } else {
            console.error(`Function ${name} not found`);
            return [id, `Function ${name} not found`];
          }
        })
      );

      const newMessages: Message[] = results.map(([id, result]: any) => ({
        role: "tool",
        content: result as string,
        tool_call_id: id as string,
      }));

      await chatWithLLMStream(newMessages, partialCallback, () => {
        endResolve();
        endCallback();
      });
      return;
    } else {
      endResolve();
      endCallback();
    }
  } catch (error: any) {
    console.error("Error communicating with LM Studio:", error);
    partialCallback(`Error: ${error.message}`);
    endResolve();
    endCallback();
  }

  return promise;
};

export { chatWithLLMStream, resetChatHistory };
