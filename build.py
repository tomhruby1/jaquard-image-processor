import subprocess

NAME = "jacquard_image_processor"
VERSION = "0.1"

CMD = [
    "python", "-m", "nuitka",
    "--standalone",
    "--deployment",
    "--no-pyi-file",
    "--onefile",
    f"--file-version={VERSION}",
    f"--output-dir={NAME}.dist",
    f"--product-name={NAME}",
    "--enable-plugin=pyside6",
    "gui.py"
]

print(f"running:\n{' '.join(CMD)}")
subprocess.run(" ".join(CMD), shell=True)