import os
import unicodedata
from PIL import Image, ImageChops, ImageDraw, ImageFont

# numpy, cv2 and cairosvg are intentionally not imported: together they cost
# ~50MB of RAM on a 512MB Pi. RGB565 packing uses Pillow lookup tables, and
# emoji are served from pre-rendered PNGs (cairosvg is only loaded lazily for
# an emoji that has no PNG yet).

class ColorUtils:
  @staticmethod
  def rgb565_to_rgb255(color_565):
    """将 RGB565 颜色值转换为 (R, G, B) 元组，每个分量范围为 0-255。"""
    red_5bit = (color_565 >> 11) & 0x1F
    green_6bit = (color_565 >> 5) & 0x3F
    blue_5bit = color_565 & 0x1F
    red_8bit = (red_5bit * 255) // 31
    green_8bit = (green_6bit * 255) // 63
    blue_8bit = (blue_5bit * 255) // 31
    return (red_8bit, green_8bit, blue_8bit)

  @staticmethod
  def hex_to_rgb255(hex_color):
    """将十六进制颜色代码转换为 (R, G, B) 元组，每个分量范围为 0-255。"""
    hex_color = hex_color.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in hex_color):
      return None
    if len(hex_color) == 6:
      r = int(hex_color[0:2], 16)
      g = int(hex_color[2:4], 16)
      b = int(hex_color[4:6], 16)
      return (r, g, b)
    elif len(hex_color) == 8:
      r = int(hex_color[0:2], 16)
      g = int(hex_color[2:4], 16)
      b = int(hex_color[4:6], 16)
      return (r, g, b)
    else:
      return None

  @staticmethod
  def get_rgb255_from_any(rgb_led):
    """自动检测输入格式并转换为 RGB (0-255) 元组。"""
    if isinstance(rgb_led, int):
      if 0 <= rgb_led <= 0xFFFF:
        return ColorUtils.rgb565_to_rgb255(rgb_led)
      else:
        return None
    elif isinstance(rgb_led, str):
      hex_color = rgb_led.lstrip("#")
      if all(c in "0123456789abcdefABCDEF" for c in hex_color) and len(hex_color) in [6, 8]:
        return ColorUtils.hex_to_rgb255(rgb_led)
      else:
        return None
    else:
      return None
  
  @staticmethod
  def calculate_luminance(rgb_tuple):
    """计算 RGB 颜色的亮度。"""
    if rgb_tuple is None:
        return -1 # 或者其他表示无效的值
    r, g, b = rgb_tuple
    return 0.299 * r + 0.587 * g + 0.114 * b


# lookup tables splitting RGB888 into the two bytes of big-endian RGB565:
# high = RRRRRGGG, low = GGGBBBBB (the bit fields never overlap, so add == or)
_R_HI = [v & 0xF8 for v in range(256)]
_G_HI = [v >> 5 for v in range(256)]
_G_LO = [(v << 3) & 0xE0 for v in range(256)]
_B_LO = [v >> 3 for v in range(256)]

class ImageUtils:
  @staticmethod
  def image_to_rgb565(image: Image.Image, width: int, height: int) -> bytes:
    image = image.convert("RGB")
    if image.size != (width, height):
      image.thumbnail((width, height), Image.LANCZOS)
      bg = Image.new("RGB", (width, height), (0, 0, 0))
      bg.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
      image = bg
    r, g, b = image.split()
    high = ImageChops.add(r.point(_R_HI), g.point(_G_HI))
    low = ImageChops.add(g.point(_G_LO), b.point(_B_LO))
    # "LA" raw bytes interleave the two channels: high, low, high, low, ...
    return Image.merge("LA", (high, low)).tobytes()

  @staticmethod
  def crop_center(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2
    return image.crop((left, top, right, bottom)).resize((target_width, target_height), Image.LANCZOS)


class EmojiUtils:
  @staticmethod
  def emoji_to_filename(char):
    return '-'.join(f"{ord(c):x}" for c in char) + ".svg"

  @staticmethod
  def render_svg_to_png(svg_path, png_path, size):
    import cairosvg  # lazy: only needed for emoji without a pre-rendered PNG
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=size, output_height=size)

  @staticmethod
  def get_local_emoji_svg_image(char, size):
    filename = EmojiUtils.emoji_to_filename(char)
    png_path = os.path.join("emoji_png", str(size), filename[:-4] + ".png")
    if not os.path.exists(png_path):
      svg_path = os.path.join("emoji_svg", filename)
      if not os.path.exists(svg_path):
        return None
      try:
        EmojiUtils.render_svg_to_png(svg_path, png_path, size)
      except Exception as e:
        print(f"[Emoji] Failed to render {svg_path}: {e}")
        return None
    try:
      return Image.open(png_path).convert("RGBA")
    except Exception as e:
      print(f"[Emoji] Failed to load {png_path}: {e}")
      return None

  @staticmethod
  def prerender_all(sizes):
    """Render every emoji SVG to PNG at the given sizes (run once, offline)."""
    for name in sorted(os.listdir("emoji_svg")):
      if not name.endswith(".svg"):
        continue
      for size in sizes:
        png_path = os.path.join("emoji_png", str(size), name[:-4] + ".png")
        if not os.path.exists(png_path):
          EmojiUtils.render_svg_to_png(os.path.join("emoji_svg", name), png_path, size)

  @staticmethod
  def is_emoji(char):
    return unicodedata.category(char) in ('So', 'Sk') or ord(char) > 0x1F000


char_size_cache = {}
line_image_cache = {}

class TextUtils:
  
  @staticmethod
  def get_char_size(font, char):
    global char_size_cache
    cache_key = (font.getname(), font.size, char)
    if cache_key in char_size_cache:
      return char_size_cache[cache_key]
    """获取字符的大小，返回宽度和高度。"""
    if EmojiUtils.is_emoji(char):
      emoji_img = EmojiUtils.get_local_emoji_svg_image(char, size=font.size)
      if emoji_img:
        char_size_cache[cache_key] = (emoji_img.width, emoji_img.height)
        return emoji_img.width, emoji_img.height
    else:
      bbox = font.getbbox(char)
      char_size_cache[cache_key] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
      return char_size_cache[cache_key]
    return 0, 0
  
  @staticmethod
  def draw_mixed_text(draw, image, text, font, start_xy):
    x, y = start_xy
    add_img = TextUtils.get_line_img(text, font)
    image.paste(add_img, (x, y), add_img)
        
  @staticmethod
  def get_line_img(text, font):
    cache_key = (font.getname(), font.size, text)
    if cache_key in line_image_cache:
      return line_image_cache[cache_key]
    x, y = 0, 0
    ascent, descent = font.getmetrics()
    baseline = y + ascent
    line_height = ascent + descent
    width = 0
    for char in text:
      width += TextUtils.get_char_size(font, char)[0]
    img = Image.new("RGBA", (width, line_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for char in text:
      if EmojiUtils.is_emoji(char):
        emoji_img = EmojiUtils.get_local_emoji_svg_image(char, size=font.size)
        if emoji_img:
          emoji_y = baseline - emoji_img.height
          img.paste(emoji_img, (x, emoji_y), emoji_img)
          x += emoji_img.width
      else:
        draw.text((x, y), char, font=font, fill=(255, 255, 255))
        char_width = TextUtils.get_char_size(font, char)[0]
        x += char_width
    line_image_cache[cache_key] = img
    return line_image_cache[cache_key]
  
  @staticmethod
  def clean_line_image_cache():
    """清除行图像缓存。"""
    global line_image_cache
    line_image_cache = {}

  @staticmethod
  def get_text_size(text, font):
    """获取文本的宽度和高度。"""
    lines = TextUtils.wrap_text(None, text, font, float('inf'))
    width = max(TextUtils.get_line_img(line, font).width for line in lines)
    height = sum(TextUtils.get_line_img(line, font).height for line in lines)
    return width, height

  @staticmethod
  def wrap_text(draw, text, font, max_width):
    """Wrap at the last space that fits; fall back to breaking between
    characters for words wider than a line (and text without spaces)."""
    lines = []
    current_line = ""
    current_width = 0
    for char in text:
      char_width = TextUtils.get_char_size(font, char)[0]
      if current_width + char_width <= max_width or not current_line:
        current_line += char
        current_width += char_width
      elif char == " ":
        # the space itself becomes the line break
        lines.append(current_line)
        current_line = ""
        current_width = 0
      elif " " in current_line:
        # move the partial word down to the next line
        split_at = current_line.rindex(" ")
        lines.append(current_line[:split_at])
        current_line = current_line[split_at + 1:] + char
        current_width = sum(TextUtils.get_char_size(font, c)[0] for c in current_line)
      else:
        lines.append(current_line)
        current_line = char
        current_width = char_width
    if current_line:
      lines.append(current_line)
    return lines