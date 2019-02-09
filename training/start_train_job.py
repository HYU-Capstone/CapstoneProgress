import ncloud_server
from ncloud_server.api.v2_api import V2Api
from ncloud_server.rest import ApiException
import os
import datetime
import time
import zmq
import paramiko
import sys


def wait_for_output(shell, ip):
    buff = ''
    while True:
        rcv = shell.recv(9999).decode('utf-8')
        buff += rcv
        print(rcv, end='')
        if 'JOB DONE' in rcv or 'JOB DONE' in buff:
            break


with open('Ncloud-GPU.pem', 'r') as fr:
    key = fr.read()

x = datetime.datetime.now()

configuration = ncloud_server.Configuration()

configuration.access_key = os.environ['NCLOUD_ACCESS_KEY']
configuration.secret_key = os.environ['NCLOUD_SECRET_KEY']
api = V2Api(ncloud_server.ApiClient(configuration))

image_product_code = 'SPSW0LINUX000029'
product_code = 'SPSVRSTAND000003'
instance_name = 'train-' + str(x.year) + '-' + str(x.month) + '-' + str(x.day)

req = ncloud_server.CreateServerInstancesRequest(
    server_image_product_code=image_product_code,
    server_product_code=product_code,
    server_description=instance_name,
    access_control_group_configuration_no_list=['82763'],
    fee_system_type_code='FXSUM',
    zone_no=3
)
try:
    res = api.create_server_instances(req)
    instance_no = res.server_instance_list[0].server_instance_no
except ApiException as e:
    print("Exception when calling V2Api->create_server_instances: %s\n" % e)
    exit()

print('Instance {} creation request done.'.format(instance_name))

ip = '0.0.0.0'
try_count = 0

while True:
    try_count += 1
    req = ncloud_server.GetServerInstanceListRequest(
        server_instance_no_list=[str(instance_no)]
    )
    try:
        res = api.get_server_instance_list(req)
        print(res.server_instance_list[0])
        if res.server_instance_list[0].server_instance_status.code != 'RUN':
            if try_count > 800:
                print('Request Timeout! Check instance status on console.')
                exit()
            if res.server_instance_list[0].server_instance_status.code == 'ST_FL':
                print('Something went wrong!')
                print('Server instance detail:')
                print(res.server_instance_list[0])
                exit()
            time.sleep(5)
        else:
            ip = res.server_instance_list[0].private_ip
            break
    except ApiException as e:
        print("Exception when calling V2Api->get_server_instance_list: %s\n" % e)

print('Instance {}({}) started with IP {}'.format(
    instance_name, instance_no, ip))

req = ncloud_server.GetRootPasswordRequest(
    private_key=key,
    server_instance_no=str(instance_no)
)

try:
    res = api.get_root_password(req)
    root_password = res.root_password
    print('Root Password:', root_password)
except ApiException as e:
    print("Exception when calling V2Api->get_root_password: %s\n" % e)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip, username='root', password=root_password)

_, stdout, _ = ssh.exec_command(
    'wget https://gist.githubusercontent.com/kyujin-cho/0b969336a46f6143f7ec71bce01eb7b1/raw/af26756e265431b3564c6283a44ce9cac57af7c3/init.sh -O init.sh')
sys.stdout.buffer.write(stdout.read())

_, stdout, _ = ssh.exec_command('chmod a+x init.sh')
sys.stdout.buffer.write(stdout.read())

_, stdout, _ = ssh.exec_command('./init.sh')
sys.stdout.buffer.write(stdout.read())

req = ncloud_server.StopServerInstancesRequest(
    server_instance_no_list=[str(instance_no)]
)

print('Train finished. Shutting down instance')

try:
    res = api.stop_server_instances(req)
except ApiException as e:
    print("Exception when calling V2Api->stop_server_instances: %s\n" % e)

try_count = 0

while True:
    try_count += 1
    req = ncloud_server.GetServerInstanceListRequest(
        server_instance_no_list=[str(instance_no)]
    )
    try:
        res = api.get_server_instance_list(req)
        if res.server_instance_list[0].server_instance_status.code != 'NSTOP':
            if try_count > 800:
                print('Request Timeout! Check instance status on console.')
                exit()
            time.sleep(5)
        else:
            break
    except ApiException as e:
        print("Exception when calling V2Api->get_server_instance_list: %s\n" % e)

print('Instance shut down. Terminating instance')

req = ncloud_server.TerminateServerInstancesRequest(
    server_instance_no_list=[str(instance_no)]
)

try:
    res = api.terminate_server_instances(req)
except ApiException as e:
    print("Exception when calling V2Api->terminate_server_instances: %s\n" % e)
    exit()

print('All job finished')
