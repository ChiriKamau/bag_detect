import serial
import time

ports = ['/dev/serial0', '/dev/ttyAMA0', '/dev/ttyS0']
bauds = [115200, 9600]

for port in ports:
    for baud in bauds:
        try:
            s = serial.Serial(port, baud, timeout=2)
            s.write(b'AT\r\n')
            time.sleep(1)
            response = s.read_all()
            print(f"{port} @ {baud}: '{response}'")
            s.close()
        except Exception as e:
            print(f"{port} @ {baud}: ERROR - {e}")