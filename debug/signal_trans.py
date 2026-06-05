import serial
import time
import sys

# 配置串口参数（根据你的实际端口修改）
ARDUINO_PORT = 'COM7'      # Windows 示例，Linux 如 '/dev/ttyUSB0'
BAUDRATE = 9600

def setup_serial():
    """建立串口连接"""
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
        time.sleep(2)       # 等待 Arduino 复位
        print(f"已连接到 {ARDUINO_PORT}")
        return ser
    except (IOError, OSError) as e:
        print(f"串口连接失败: {e}")
        sys.exit(1)

def send_angles(ser, pan, tilt):
    """发送水平、俯仰角度（支持小数）"""
    cmd = f"{pan},{tilt}\n"
    ser.write(cmd.encode())
    print(f"发送: {cmd.strip()}")
    
    # 可选：读取 Arduino 的返回信息（如 "Pan: 45.00, Tilt: 30.00"）
    # 注意：Arduino 代码中每次运动后会打印当前角度，这里读取一行
    # 如果不需要读取，可以注释掉下面的循环
    if ser.in_waiting:
        response = ser.readline().decode().strip()
        print(f"Arduino 回复: {response}")

def main():
    ser = setup_serial()
    print("请输入角度，格式：水平 俯仰 (例如 45 30) 或输入 q 退出")
    
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() == 'q':
            break
        parts = user_input.split()
        if len(parts) == 2:
            try:
                pan = float(parts[0])
                tilt = float(parts[1])
                send_angles(ser, pan, tilt)
            except ValueError:
                print("请输入数字，例如 45 30")
        else:
            print("格式错误，请用空格分隔两个角度，例如 45 30")
    
    ser.close()
    print("串口已关闭")

if __name__ == "__main__":
    main()