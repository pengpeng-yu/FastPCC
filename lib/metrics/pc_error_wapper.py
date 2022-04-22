import subprocess


def mpeg_pc_error(infile1, infile2, resolution, normal=False, color=False, command='pc_error_d', threads=1):
    # https://github.com/NJUVISION/PCGCv2
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)",
                "h.       1(p2point)", "h.,PSNR  1(p2point)"]
    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)",
                "h.       2(p2point)", "h.,PSNR  2(p2point)"]
    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)",
                "h.        (p2point)", "h.,PSNR   (p2point)"]
    headers = headers1 + headers2 + headersF
    command = f'{command}' \
              f' -a {infile1}' \
              f' -b {infile2}' \
              f' --hausdorff=1' \
              f' --resolution={resolution - 1}' \
              f' --nbThreads={threads}'

    if normal:
        headers_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                           "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                           "mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        headers += headers_p2plane
        command = command + ' -n ' + infile1
    if color:
        headers_color_f = ['c[0],    F', 'c[1],    F', 'c[2],    F',
                           'c[0],PSNRF', 'c[1],PSNRF', 'c[2],PSNRF']
        headers += headers_color_f
        command = command + ' --color=1'

    results = {}
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        for key in headers:
            if key in line:
                line = line.strip()
                value = float(line[line.find(': ') + 2:])
                results[key] = value
        c = subp.stdout.readline()
    return results
