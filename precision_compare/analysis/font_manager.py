import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def setup_chinese_font():
    """设置支持中文和负号的字体"""
    # 尝试使用系统中文字体，优先使用更完整的字体
    font_paths = [
        "C:/Windows/Fonts/Microsoft YaHei UI/msyh.ttc",  # Windows 微软雅黑
        "C:/Windows/Fonts/Microsoft YaHei/msyh.ttc",  # Windows 微软雅黑 (备选路径)
        "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑 (备选路径)
        "C:/Windows/Fonts/SimSun.ttc",  # Windows 宋体
        "C:/Windows/Fonts/SimHei.ttf",  # Windows 黑体
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
    ]

    font = None
    for font_path in font_paths:
        try:
            if not Path(font_path).exists():
                continue

            font = FontProperties(fname=font_path)
            # 测试字体是否支持负号
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "-1.0", fontproperties=font)
            plt.close(fig)
            logger.info(f"成功加载字体: {font_path}")
            break
        except Exception as e:
            logger.warning(f"字体 {font_path} 加载失败: {str(e)}")
            continue

    if font is not None:
        # 设置全局字体
        plt.rcParams["font.family"] = [font.get_name(), "sans-serif"]

        # 设置数学字体，确保负号正确显示
        plt.rcParams["axes.unicode_minus"] = False  # 使用ASCII的减号
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = font.get_name()
        plt.rcParams["mathtext.it"] = font.get_name()
        plt.rcParams["mathtext.bf"] = font.get_name()

        return True
    else:
        logger.warning("未找到合适的中文字体，图表中的中文和负号可能无法正确显示")
        return False
