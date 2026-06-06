"""
电机控制模块
通过串口与Arduino通信，控制云台/电机。
支持三种模式:
    模式0 - 复位归零
    模式1 - 开环位置控制（绝对角度）
    模式2 - 闭环速度控制（度/秒）
使用方法:
    from motor_control import MotorController
    mc = MotorController(port='COM7')
    mc.reset()              # 复位
    mc.set_position(45, -10)  # 水平45度，俯仰-10度
    mc.set_velocity(30, -20)  # 水平30度/秒，俯仰-20度/秒
"""

import serial
import time


class MotorController:
    """云台电机串口控制器"""

    def __init__(self, port='COM7', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        """建立串口连接"""
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        print(f"串口已连接: {self.port} @ {self.baudrate}")
        return self.ser

    def disconnect(self):
        """关闭串口连接"""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("串口已断开")

    def _send_command(self, cmd_str):
        """发送命令字符串并读取响应"""
        self.ser.write((cmd_str + "\n").encode())
        print(f"发送: {cmd_str}")
        time.sleep(0.1)
        responses = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode().strip()
            if line:
                print("Arduino:", line)
                responses.append(line)
        return responses

    def reset(self):
        """模式0: 复位归零"""
        if self.ser is None:
            raise RuntimeError("串口未连接，请先调用 connect()")
        return self._send_command("0")

    def set_position(self, pan_angle, tilt_angle):
        """模式1: 开环位置控制（绝对角度）"""
        if self.ser is None:
            raise RuntimeError("串口未连接，请先调用 connect()")
        return self._send_command(f"1,{pan_angle},{tilt_angle}")
    
    def set_velocity(self, pan_speed, tilt_speed):
        """模式2: 闭环速度控制（度/秒）"""
        if self.ser is None:
            raise RuntimeError("串口未连接，请先调用 connect()")
        return self._send_command(f"2,{pan_speed},{tilt_speed}")

    def send_raw(self, mode, arg1=None, arg2=None):
        """通用发送接口"""
        if mode == 0:
            return self.reset()
        elif mode == 1:
            return self.set_position(arg1, arg2)
        elif mode == 2:
            return self.set_velocity(arg1, arg2)
        else:
            raise ValueError(f"无效模式: {mode}，有效模式为 0, 1, 2")


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    mc = MotorController(port='COM7', baudrate=9600)
    mc.connect()

    print("\n=== 控制模式 ===")
    print("0          -> 复位（重新归零）")
    print("1 水平 俯仰 -> 开环位置控制（绝对角度）")
    print("2 水平速度 俯仰速度 -> 闭环速度控制（度/秒）")
    print("示例: 1 45 -10")
    print("     2 30 -20")
    print("     q 退出")

    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() == 'q':
            break
        parts = user_input.split()
        if not parts:
            continue
        try:
            mode = int(parts[0])
            if mode == 0:
                mc.reset()
            elif mode == 1:
                if len(parts) >= 3:
                    mc.set_position(float(parts[1]), float(parts[2]))
                else:
                    print("错误：模式1需要提供水平和俯仰角度")
            elif mode == 2:
                if len(parts) >= 3:
                    mc.set_velocity(float(parts[1]), float(parts[2]))
                else:
                    print("错误：模式2需要提供水平和俯仰速度（度/秒）")
            else:
                print("无效模式，请输入 0, 1, 2")
        except ValueError as e:
            print(f"输入格式错误: {e}")

    mc.disconnect()
    print("已退出")
