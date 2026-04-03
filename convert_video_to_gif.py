#!/usr/bin/env python3
"""
将 MP4 视频转换为 GIF 动图
优化文件大小：降低帧率和分辨率
"""

import sys
import os

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("❌ 缺少依赖: moviepy")
    print("请安装: pip install moviepy")
    sys.exit(1)

def convert_to_gif(input_file, output_file=None, fps=10, scale=0.5):
    """
    转换视频为 GIF

    Args:
        input_file: 输入视频文件路径
        output_file: 输出 GIF 文件路径（可选）
        fps: 帧率（默认10，降低可减小文件）
        scale: 缩放比例（默认0.5，即50%）
    """
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 {input_file}")
        sys.exit(1)

    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '.gif'

    print(f"📹 正在转换: {input_file}")
    print(f"   帧率: {fps} fps")
    print(f"   缩放: {int(scale*100)}%")
    print(f"   输出: {output_file}")
    print()
    print("⏳ 处理中，请稍候...")

    try:
        # 加载视频
        clip = VideoFileClip(input_file)

        # 获取原始尺寸
        width, height = clip.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        print(f"   原始尺寸: {width}x{height}")
        print(f"   新尺寸: {new_width}x{new_height}")
        print(f"   视频时长: {clip.duration:.1f} 秒")

        # 调整大小并转换为 GIF
        clip_resized = clip.resize((new_width, new_height))
        clip_resized.write_gif(output_file, fps=fps, program='ffmpeg')

        clip.close()

        # 显示文件大小
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print()
        print(f"✅ 转换完成!")
        print(f"   文件: {output_file}")
        print(f"   大小: {size_mb:.2f} MB")

        if size_mb > 10:
            print()
            print("⚠️  警告: GIF 文件较大 (>10MB)，可能影响加载速度")
            print("   建议: 降低参数重新转换")
            print("   python convert_video_to_gif.py --fps 8 --scale 0.4")

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='将视频转换为 GIF')
    parser.add_argument('input', nargs='?', default='assets/animation.mp4',
                        help='输入视频文件路径 (默认: assets/animation.mp4)')
    parser.add_argument('-o', '--output', help='输出 GIF 文件路径')
    parser.add_argument('--fps', type=int, default=10,
                        help='帧率 (默认: 10)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='缩放比例 (默认: 0.5 即50%%)')

    args = parser.parse_args()

    convert_to_gif(args.input, args.output, args.fps, args.scale)
