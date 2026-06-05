import serial
import time

SERIAL_PORT = 'COM7'   # 修改为实际端口
BAUDRATE = 9600

def setup_serial():
    ser = serial.Serial(SERIAL_PORT, 00BAUDRATE, timeout=1)
    time.sleep(2)
    print("串口已连接")
    return ser

def send_command(ser, mode, arg1=None, arg2=None):
    if mode == 0:
        cmd = "0\n"
    elif mode == 1:
        cmd = f"1,{arg1},{arg2}\n"
    elif mode == 2:
        cmd = f"2,{arg1},{arg2}\n"
    else:
        print("无效模式")
        return
    ser.write(cmd.encode())
    print(f"发送: {cmd.strip()}")
    time.sleep(0.1)
    while ser.in_waiting:
        line = ser.readline().decode().strip()
        if line:
            print("Arduino:", line)

def main():
    ser = setup_serial()
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
                send_command(ser, 0)
            elif mode == 1:
                if len(parts) >= 3:
                    pan = float(parts[1])
                    tilt = float(parts[2])
                    send_command(ser, 1, pan, tilt)
                else:
                    print("错误：模式1需要提供水平和俯仰角度")
            elif mode == 2:
                if len(parts) >= 3:
                    pan_speed = float(parts[1])
                    tilt_speed = float(parts[2])
                    send_command(ser, 2, pan_speed, tilt_speed)
                else:
                    print("错误：模式2需要提供水平和俯仰速度（度/秒）")
            else:
                print("无效模式，请输入 0,1,2")
        except ValueError:
            print("输入格式错误，请重新输入")
    ser.close()
    print("已退出")

if __name__ == "__main__":
    main()