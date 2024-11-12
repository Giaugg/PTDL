import pandas as pd
import re
from sklearn.linear_model import LinearRegression

# Đọc file CSV
df = pd.read_csv('IMDb_Movies_India.csv', encoding='ISO-8859-1')

pd.set_option('display.max_columns', None)

print(df.head())         # Xem 5 dòng đầu tiên của DataFrame
df.info()         # Thông tin về kiểu dữ liệu và giá trị bị thiếu
df.describe()     # Thống kê mô tả cho các cột số

######################### Lọc Dữ Liệu ################################
# Lọc các dữ liệu phân loại
categorical = df.select_dtypes(include=['object']).keys()
print(categorical)
# Lọc các dữ liệu số lượng
quantitative = df.select_dtypes(include=['int64', 'float64']).keys()
print(quantitative)


######################## XỬ LÍ DỮ LIỆU  NULL ##########################
#check null
print(df.isnull().sum())
# Hàm để chuyển đổi giá trị trong cột 'vote' tùy theo dấu phẩy và 'M'
# Hàm để chuyển đổi giá trị trong cột 'vote', loại bỏ tất cả ký tự không phải là số
def convert_vote(value):
    if isinstance(value, str):
        # Loại bỏ tất cả các ký tự không phải là số, phẩy hoặc 'M'
        value = re.sub(r'[^\d,\.M]', '', value)
        
        # Nếu có 'M', chuyển thành triệu
        if 'M' in value:
            return float(value.replace(',', '').replace('M', '')) * 1_000_000
        # Nếu có dấu phẩy (nghìn), chỉ cần loại bỏ dấu phẩy
        elif ',' in value:
            return float(value.replace(',', ''))
    # Nếu không phải chuỗi, giữ nguyên giá trị
    return value

# Áp dụng hàm convert_vote lên cột 'Votes'
df['Votes'] = df['Votes'].apply(convert_vote)
# Chuyển đổi cột 'Votes' thành số thực (float) nếu cần
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
# Điền giá trị NaN (nếu có) bằng giá trị trung bình
df['Votes'].fillna(df['Votes'].mean(), inplace=True)


#######################################################################
# Duyệt qua từng cột và điền giá trị null bằng trung vị của cột
for column in df.columns:
    # Kiểm tra nếu cột có kiểu dữ liệu số, vì trung vị chỉ áp dụng cho số
    if df[column].dtype in ['float64', 'int64']:
        median_value = df[column].median()  # Tính trung vị
        df[column].fillna(median_value, inplace=True)  # Điền giá trị null bằng trung vị


#######################################################################
# Chuyển đạo diễn với diễn viên null thành Unknow
df['Director'].fillna('Unknown', inplace=True)
df['Actor 1'].fillna('Unknown', inplace=True)
df['Actor 2'].fillna('Unknown', inplace=True)
df['Actor 3'].fillna('Unknown', inplace=True)


#######################################################################
#Format lại cột duration và fill dữ liệu
# Bước 1: Chuyển đổi `Duration` sang số nguyên
df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)  # Chuyển sang float để tránh lỗi nếu có NaN
# Bước 2: Tính trung vị của cột `Duration` (loại bỏ NaN)
median_duration = df['Duration'].median()
# Bước 3: Điền các giá trị NaN bằng trung vị
df['Duration'].fillna(median_duration, inplace=True)
#Fill Genre
df['Genre'].fillna('Unknown', inplace=True)


#######################################################################
# In lại DataFrame để kiểm tra
print(df.head())

###################### Xử lí Genre ##################################
# Giả sử cột 'Genre' chứa các thể loại
df['Genre'] = df['Genre'].fillna('Unknown')  # Điền giá trị thiếu

# Tính toán tần suất xuất hiện của mỗi thể loại
genre_counts = df['Genre'].value_counts()

# Xác định ngưỡng cho thể loại ít gặp (ví dụ: thể loại xuất hiện dưới 5 lần)
threshold = 10

# Tạo cột mới 'Simplified_Genre', thay thế các thể loại ít gặp thành 'Tổng hợp'
df['Simplified_Genre'] = df['Genre'].apply(lambda x: x if genre_counts[x] >= threshold else 'Other')

# Kiểm tra kết quả
genre_counts = df['Genre'].value_counts()
print(genre_counts.head())


