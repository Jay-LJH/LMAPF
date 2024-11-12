import argparse
import subprocess
import sys
import os
import shutil
import platform
import ctypes

class PythonProfiler:
    def __init__(self, duration=30, output="flamegraph.svg"):
        self.duration = duration
        self.output = output
        self._check_permissions()
    
    def _is_admin(self):
        """检查是否具有管理员权限"""
        try:
            if platform.system() == 'Windows':
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False
            
    def _check_permissions(self):
        """检查并处理权限问题"""
        if not self._is_admin():
            print("需要提升权限才能运行性能分析。")
            print("\n请使用以下方式运行:")
            if platform.system() == 'Windows':
                print("1. 以管理员身份打开命令提示符")
                print("2. 然后运行此脚本")
                print("\n或者右键点击 Python/命令提示符，选择'以管理员身份运行'")
            else:
                current_command = ' '.join(sys.argv)
                print(f"\nsudo env \"PATH=$PATH\" python {current_command}")
            sys.exit(1)
    
    def _run_with_sudo(self, cmd):
        """使用sudo运行命令"""
        if platform.system() != 'Windows' and not self._is_admin():
            # 添加 sudo 和环境变量
            sudo_cmd = ['sudo', 'env', f"PATH={os.getenv('PATH')}"]
            cmd = sudo_cmd + cmd
        return cmd
            
    def profile_script(self, script_path, script_args=None):
        """分析Python脚本"""
        if not os.path.exists(script_path):
            print(f"错误: 脚本文件 {script_path} 不存在")
            return False
            
        cmd = [
            "py-spy", "record",
            "-d", str(self.duration),
            "-o", self.output,
            "--format", "flamegraph",
            "--nonblocking",
            "--", sys.executable, script_path
        ]
        
        if script_args:
            cmd.extend(script_args)
            
        # 使用sudo运行命令
        cmd = self._run_with_sudo(cmd)
            
        try:
            print(f"开始分析脚本: {script_path}")
            print(f"采样时长: {self.duration}秒")
            subprocess.run(cmd, check=True)
            print(f"\n火焰图已生成: {self.output}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"错误: 分析失败 - {str(e)}")
            if "Permission denied" in str(e):
                print("\n权限被拒绝。请使用管理员权限运行。")
                if platform.system() != 'Windows':
                    print(f"\n运行: sudo env \"PATH=$PATH\" python {' '.join(sys.argv)}")
            return False
            
    def profile_process(self, pid):
        """分析运行中的Python进程"""
        cmd = [
            "py-spy", "record",
            "-d", str(self.duration),
            "-o", self.output,
            "--format", "flamegraph",
            "--pid", str(pid)
        ]
        
        # 使用sudo运行命令
        cmd = self._run_with_sudo(cmd)
        
        try:
            print(f"开始分析进程 PID: {pid}")
            print(f"采样时长: {self.duration}秒")
            subprocess.run(cmd, check=True)
            print(f"\n火焰图已生成: {self.output}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"错误: 分析失败 - {str(e)}")
            if "Permission denied" in str(e):
                print("\n权限被拒绝。请使用管理员权限运行。")
                if platform.system() != 'Windows':
                    print(f"\n运行: sudo env \"PATH=$PATH\" python {' '.join(sys.argv)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Python程序性能分析工具')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Python脚本文件路径')
    group.add_argument('-p', '--pid', type=int, help='运行中的Python进程ID')
    parser.add_argument('-d', '--duration', type=int, default=30, help='采样时长(秒)')
    parser.add_argument('-o', '--output', default='flamegraph.svg', help='输出SVG文件路径')
    parser.add_argument('script_args', nargs='*', help='传递给Python脚本的参数')
    
    args = parser.parse_args()
    
    profiler = PythonProfiler(duration=args.duration, output=args.output)
    
    if args.file:
        profiler.profile_script(args.file, args.script_args)
    else:
        profiler.profile_process(args.pid)

if __name__ == '__main__':
    main()