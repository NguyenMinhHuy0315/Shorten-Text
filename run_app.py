import os
import subprocess
import sys

# Lấy đường dẫn thư mục chứa file run_app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Xây dựng đường dẫn tuyệt đối đến file app.py
# Đảm bảo rằng đường dẫn này hoạt động bất kể file run_app.py được đặt ở đâu
app_path = os.path.join(current_dir, "project eng-vie gg trans copy", "app.py")

# Kiểm tra xem file app.py có tồn tại không
if not os.path.exists(app_path):
    print(f"Lỗi: File '{app_path}' không tồn tại.")
    sys.exit(1)

# Chạy file app.py
try:
    # Sử dụng subprocess để chạy file app.py bằng Python
    subprocess.run(["python", app_path], check=True)
except subprocess.CalledProcessError as e:
    # Xử lý lỗi nếu quá trình chạy file app.py thất bại
    print(f"Lỗi: Không thể chạy file '{app_path}'.\n{e}")
    sys.exit(1)