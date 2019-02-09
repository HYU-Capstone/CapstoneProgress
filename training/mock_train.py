import time
import netifaces as ni

time.sleep(15)
ni.ifaddresses('eth0')
ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']
print(ip + ' JOB DONE')
