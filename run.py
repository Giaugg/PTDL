import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Đọc file CSV
df = pd.read_csv('imdb_top_2000_movies.csv', encoding='ISO-8859-1')
pd.set_option('display.max_columns', None)

# print(df.head())         # Xem 5 dòng đầu tiên của DataFrame
# df.info()         # Thông tin về kiểu dữ liệu và giá trị bị thiếu


# ######################### Lọc Dữ Liệu ################################
# Lọc các dữ liệu phân loại
categorical = df.select_dtypes(include=['object']).keys()
# print(categorical)
# Lọc các dữ liệu số lượng
quantitative = df.select_dtypes(include=['int64', 'float64']).keys()
# print(quantitative)

# Hàm kiểm tra dữ liệu không đúng
def detect_invalid_data(df):
    invalid_data = {}

    # Kiểm tra cột Release Year
    invalid_year = df[~df['Release Year'].apply(lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()))]
    if not invalid_year.empty:
        invalid_data['Release Year'] = invalid_year[['Release Year']]

    # Kiểm tra cột Duration
    invalid_duration = df[~df['Duration'].apply(lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()))]
    if not invalid_duration.empty:
        invalid_data['Duration'] = invalid_duration[['Duration']]

    # Kiểm tra cột IMDB Rating
    invalid_rating = df[~df['IMDB Rating'].apply(lambda x: isinstance(x, float) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()))]
    if not invalid_rating.empty:
        invalid_data['IMDB Rating'] = invalid_rating[['IMDB Rating']]

    # Kiểm tra cột Metascore
    invalid_metascore = df[~df['Metascore'].apply(lambda x: isinstance(x, int) or (isinstance(x, float) and x.is_integer()) or (isinstance(x, str) and x.replace('.', '', 1).isdigit() and float(x).is_integer()))]
    if not invalid_metascore.empty:
        invalid_data['Metascore'] = invalid_metascore[['Metascore']]

    # Kiểm tra cột Votes
    invalid_vote = df[~df['Votes'].apply(lambda x: isinstance(x, str) and x.replace(',', '').isdigit())]
    if not invalid_vote.empty:
        invalid_data['Votes'] = invalid_vote[['Votes']]
    else:
        # Chuyển đổi Votes thành int
        df['Votes'] = df['Votes'].apply(lambda x: float(x.replace(',', '')))

    # Kiểm tra cột Gross
    invalid_gross = df[~df['Gross'].apply(lambda x: isinstance(x, str) and x.startswith('$') and x.endswith('M') and x[1:-1].replace('.', '', 1).isdigit())]
    if not invalid_gross.empty:
        invalid_data['Gross'] = invalid_gross[['Gross']]
    else:
        # Chuyển đổi Gross thành float
        df['Gross'] = df['Gross'].apply(lambda x: float(x[1:-1]) * 1e6)

    return invalid_data


# Phát hiện dữ liệu không đúng
invalid_data = detect_invalid_data(df)

# In ra các dữ liệu không đúng
for column, data in invalid_data.items():
    print(f"Invalid data in column {column}:")
    print(data.to_string(index=False))
# df.info()         # Thông tin về kiểu dữ liệu và giá trị bị thiếu

# #####################################################################
                # 2.1.1. Xóa dữ liệu không hợp lệ
# Loại bỏ các giá trị không phải số trong cột Release Year và điền lại các giá trị thiếu
df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')
df['Release Year'].fillna(df['Release Year'].median(), inplace=True)

# Chuyển đổi cột 'Release Year' thành kiểu int
df['Release Year'] = df['Release Year'].astype(int)
# #####################################################################
# 2.1.2. Chuẩn hóa dữ liệu
# Loại bỏ ký hiệu $ và M, chuyển thành số thập phân 
df['Gross'] = df['Gross'].str.replace('[\$,M]', '', regex=True).astype(float) 
# Chuyển đổi từ triệu đô la sang đô la (nếu cần) d
df['Gross'] = df['Gross'] * 1e6 
# Chuẩn hóa cột Gross (đơn vị triệu USD)
df['Gross'] = df['Gross'] / 1e6

# Chuyển đổi cột Votes thành số
df['Votes'] = df['Votes'].replace(',', '', regex=True).astype(float)
# Chuẩn hóa cột Votes (đơn vị triệu phiếu)
df['Votes'] = df['Votes'] / 1e6
# Chuyển đổi các cột phân loại thành số (nếu cần)
df = pd.get_dummies(df, columns=['Genre', 'Director', 'Cast'], drop_first=True)
# #####################################################################

# 2.1.3. Xử lí các dữ liệu còn thiếu
# Xử lý giá trị Na 
df['Gross'].fillna(df['Gross'].median(), inplace=True)
# print(df.isnull().sum())
# Fill cột Metascore bằng int thay vì float
df['Metascore'].fillna(int(round(df['Metascore'].mean())), inplace=True)
df['Metascore'] = df['Metascore'].astype(int)


# # Phát hiện dữ liệu không đúng
# invalid_data = detect_invalid_data(df)

# # In ra các dữ liệu không đúng
# for column, data in invalid_data.items():
#     print(f"Invalid data in column {column}:")
#     # print(data.to_string(index=False))
# df.info()         # Thông tin về kiểu dữ liệu và giá trị bị thiếu

#####################################################################
# Chia dữ liệu thành các biến độc lập (X) và biến phụ thuộc (y)
X = df[['Metascore']]
y = df['IMDB Rating']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phân tích hồi quy tuyến tính đơn biến
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)
print("Linear Regression - MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression - R2 Score:", r2_score(y_test, y_pred_linear))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Metascore')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích hồi quy đa thức
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
y_pred_poly = poly_regressor.predict(X_poly_test)
print("Polynomial Regression - MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression - R2 Score:", r2_score(y_test, y_pred_poly))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_poly, color='red', label='Predicted')
plt.title('Polynomial Regression')
plt.xlabel('Metascore')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng cây quyết định
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
print("Decision Tree - MSE:", mean_squared_error(y_test, y_pred_tree))
print("Decision Tree - R2 Score:", r2_score(y_test, y_pred_tree))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_tree, color='red', label='Predicted')
plt.title('Decision Tree Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng rừng ngẫu nhiên
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)
print("Random Forest - MSE:", mean_squared_error(y_test, y_pred_forest))
print("Random Forest - R2 Score:", r2_score(y_test, y_pred_forest))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_forest, color='red', label='Predicted')
plt.title('Random Forest Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng K láng giềng gần nhất
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)
y_pred_knn = knn_regressor.predict(X_test)
print("K-Nearest Neighbors - MSE:", mean_squared_error(y_test, y_pred_knn))
print("K-Nearest Neighbors - R2 Score:", r2_score(y_test, y_pred_knn))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_knn, color='red', label='Predicted')
plt.title('K-Nearest Neighbors Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng máy vector hỗ trợ
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X_train[['Metascore']], y_train)
y_pred_svr = svr_regressor.predict(X_test[['Metascore']])
print("Support Vector Regression - MSE:", mean_squared_error(y_test, y_pred_svr))
print("Support Vector Regression - R2 Score:", r2_score(y_test, y_pred_svr))

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Metascore'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Metascore'], y_pred_svr, color='red', label='Predicted')
plt.title('Support Vector Regression')
plt.xlabel('Metascore')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()



X = df[['Votes']]
y = df['IMDB Rating']
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phân tích hồi quy tuyến tính đơn biến
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)
print("Linear Regression - MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression - R2 Score:", r2_score(y_test, y_pred_linear))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Votes')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích hồi quy đa thức
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
y_pred_poly = poly_regressor.predict(X_poly_test)
print("Polynomial Regression - MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression - R2 Score:", r2_score(y_test, y_pred_poly))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_poly, color='red', label='Predicted')
plt.title('Polynomial Regression')
plt.xlabel('Votes')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng cây quyết định
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
print("Decision Tree - MSE:", mean_squared_error(y_test, y_pred_tree))
print("Decision Tree - R2 Score:", r2_score(y_test, y_pred_tree))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_tree, color='red', label='Predicted')
plt.title('Decision Tree Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng rừng ngẫu nhiên
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)
print("Random Forest - MSE:", mean_squared_error(y_test, y_pred_forest))
print("Random Forest - R2 Score:", r2_score(y_test, y_pred_forest))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_forest, color='red', label='Predicted')
plt.title('Random Forest Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng K láng giềng gần nhất
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)
y_pred_knn = knn_regressor.predict(X_test)
print("K-Nearest Neighbors - MSE:", mean_squared_error(y_test, y_pred_knn))
print("K-Nearest Neighbors - R2 Score:", r2_score(y_test, y_pred_knn))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_knn, color='red', label='Predicted')
plt.title('K-Nearest Neighbors Regression')
plt.xlabel('Sample Index')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()

# Phân tích sử dụng máy vector hỗ trợ
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X_train[['Votes']], y_train)
y_pred_svr = svr_regressor.predict(X_test[['Votes']])
print("Support Vector Regression - MSE:", mean_squared_error(y_test, y_pred_svr))
print("Support Vector Regression - R2 Score:", r2_score(y_test, y_pred_svr))

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Votes'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Votes'], y_pred_svr, color='red', label='Predicted')
plt.title('Support Vector Regression')
plt.xlabel('Votes')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()