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
    const chatCompletion = await openai.chat.completions.create({
      model: lmStudioModel,
      messages: messages as any,
      stream: true,
      tools: enableTools ? llmTools : undefined,
    });

    let partialAnswer = "";
    const functionCallsPackages: any[] = [];

    for await (const chunk of chatCompletion) {
      if (chunk.choices[0].delta.content) {
        partialCallback(chunk.choices[0].delta.content);
        partialAnswer += chunk.choices[0].delta.content;
      }
      if (chunk.choices[0].delta.tool_calls) {
        functionCallsPackages.push(...chunk.choices[0].delta.tool_calls);
      }
    }

    const answer = partialAnswer;
    const functionCalls = combineFunction(functionCallsPackages);

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
