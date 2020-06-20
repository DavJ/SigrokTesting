import subprocess

sigrok_cmd = 'sigrok-cli -d hantek-dso --config "samplerate=10MHz:buffersize=32768" -O analog --continuous'

proc = subprocess.Popen(sigrok_cmd, shell=True,)


while True:
    proc.communicate()
    output = proc.stdout.readline()
    print(output)

