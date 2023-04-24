import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def veriler_read():
    veriler = pd.read_csv('mobile_phone_price.csv')
    print(veriler)

    #istatistik için veri temizleme ve düzenleme
    #Storage sütununu düzenleme
    storage = veriler["Storage "].astype(str)
    # "GB" ifadesini kaldır
    for i in range(len(storage)):
            if 'GB' in storage[i]:
                storage[i] = storage[i].replace('GB', '')
            elif ' ' in storage[i]:
                storage[i] = storage[i].replace(' ', '')
    veriler['Storage '] = storage
    #RAM sütununu düzenleme
    ram = veriler["RAM "].astype(str)
    #GB ifadesini kaldir
    for i in range(len(ram)):
        if 'GB' in ram[i]:
            ram[i] = ram[i].replace('GB','')
        elif ' ' in ram[i]:
            ram[i] = ram[i].replace(' ','')

    veriler['RAM '] = ram

    screen = veriler['Screen Size (inches)'].astype(str)
    for i in range(len(screen)):
        if ' ' in screen[i]:
            screen[i] = screen[i].replace(' ','')
        elif r'\(.*\)' in screen[i]:
            screen[i] = screen[i].replace(r'\(.*\)', '').str.strip()
    veriler['Screen Size (inches)'] = screen

    veriler[['Screen Size (inches 1)', 'Screen Size (inches 2)']] = veriler['Screen Size (inches)'].str.split('+', expand=True)
    veriler['Screen Size (inches 1)'] = veriler['Screen Size (inches 1)'].str.replace(' ', '').astype(float)
    veriler['Screen Size (inches 2)'] = veriler['Screen Size (inches 2)'].str.replace(' ', '').astype(float)
    veriler['Screen Size (inches)'] = veriler.apply(
        lambda x: x['Screen Size (inches 1)'] + x['Screen Size (inches 2)'] if pd.notnull(x['Screen Size (inches 2)']) else
        x['Screen Size (inches 1)'], axis=1)
    # 'Screen Size (inches)' sütunu siliniyor
    veriler.drop('Screen Size (inches)', axis=1, inplace=True)

    # 'Camera (MP)' sütunu ayrıştırılıyor
    kamera = veriler['Camera (MP)'].str.split('+', expand=True)
    kamera.columns = [f"Camera {i+1}" for i in range(kamera.shape[1])]
    print(kamera.columns)

    # Kameranın megapiksel değerleri ayrıştırılıyor ve eksik değerler NaN ile değiştiriliyor
    for col in kamera.columns:
        kamera[col] = kamera[col].str.strip().str.replace('MP','')
        kamera[col] = pd.to_numeric(kamera[col], errors='coerce')

    # 'Camera (MP)' sütunu siliniyor ve yeni sütunlar ekleniyor
    veriler.drop('Camera (MP)', axis=1, inplace=True)
    veriler = pd.concat([veriler, kamera], axis=1)

    veriler['Price ($)'] = veriler['Price ($)'].str.replace('$', '')  # $ işareti kaldırılır
    veriler['Price ($)'] = veriler['Price ($)'].str.replace(',', '').astype(float)  # Virgül ile ayrılmış binlik haneler kaldırılır ve float veri tipine dönüştürülür
    veriler = veriler.rename(columns={'Price ($)': 'Price'})  # Sütun adı 'Price' olarak değiştirilir

    veriler.fillna(0, inplace=True)
    # Verileri tablo halinde düzenleme
    print(veriler.to_string(index=False))

    le = LabelEncoder()
    veriler['Brand'] = le.fit_transform(veriler['Brand'])
    num_to_cat = dict(zip(le.transform(le.classes_), le.classes_))
    print(num_to_cat)
    veriler['Brand'] = veriler['Brand'].astype('category')
    veriler['Model'] = veriler['Model'].astype('category')
    veriler['Brand'] = veriler['Brand'].cat.codes
    veriler['Model'] = veriler['Model'].cat.codes



    # Verileri tablo halinde düzenleme
    print(veriler.to_string(index=False))

    #veri sutün analizi ve istatistiği
    print("Brand: ", veriler["Brand"].nunique(), " farklı marka var.")
    print("Model: ", veriler["Model"].nunique(), " farklı model var.")
    print("Storage: Ortalama=", veriler["Storage "].astype(int).mean(), "GB, Standart Sapma=", veriler["Storage "].astype(int).std(), "GB.")
    print("RAM: Ortalama=", veriler["RAM "].astype(int).mean(), "GB, Standart Sapma=", veriler["RAM "].astype(int).std(), "GB.")
    print("Screen Size (inches 1) : Ortalama=", veriler["Screen Size (inches 1)"].astype(float).mean(), "inç, Standart Sapma=", veriler["Screen Size (inches 1)"].astype(float).std(), "inç.")
    print("Screen Size (inches 2): Ortalama=", veriler["Screen Size (inches 2)"].astype(float).mean(), "inç, Standart Sapma=", veriler["Screen Size (inches 2)"].astype(float).std(), "inç.")
    print("Camera (MP) 1: Ortalama=", veriler["Camera 1"].mean(), "MP, Standart Sapma=", veriler["Camera 1"].std(), "MP.")
    print("Camera (MP) 2: Ortalama=", veriler["Camera 2"].mean(), "MP, Standart Sapma=", veriler["Camera 2"].std(), "MP.")
    print("Camera (MP) 3: Ortalama=", veriler["Camera 3"].mean(), "MP, Standart Sapma=", veriler["Camera 3"].std(), "MP.")
    print("Camera (MP) 4: Ortalama=", veriler["Camera 4"].mean(), "MP, Standart Sapma=", veriler["Camera 4"].std(), "MP.")
    print("Battery Capacity (mAh): Ortalama=", veriler["Battery Capacity (mAh)"].mean(), "mAh, Standart Sapma=", veriler["Battery Capacity (mAh)"].std(), "mAh.")
    print("Price ($): Ortalama=", veriler["Price"].mean(), "$, Standart Sapma=", veriler["Price"].std(), "$.")
    Linear_Regression(veriler)
    Brand_Lojistik_Regresyon(veriler)
    PCA_data(veriler)


#-------DOGRUSAL REGRESYON-----------------

def Linear_Regression(veriler):
    # Özellikleri ve hedef değişkeni ayırma
    X = veriler.drop('Price', axis=1)
    y = veriler['Price']

    # Eğitim ve test setlerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Doğrusal regresyon modelini oluşturma ve eğitim
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_pred = lr_model.predict(X_test)

    # Model performansını değerlendirme
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Test hatası (RMSE): {:.2f}".format(rmse))


    # Test seti üzerinde tahmin yapma
    y_pred = lr_model.predict(X_test)

    # Tahmin sonuçlarını yazdırma
    print('Tahmin edilen fiyatlar: ', y_pred)
    print('Gerçek fiyatlar: ', y_test)

    # Model performansını değerlendirme
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print('Model performansı (R^2): ', r2)   #1'e ne kadar yakınsa o kadar iyi


    # Özellikleri ve hedef değişkeni ayırma
    X = veriler.drop('Price', axis=1)
    y = veriler['Price']

    # Eğitim ve test setlerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Doğrusal regresyon modelini oluşturma ve eğitim
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Tahmini fiyat ile gerçek fiyat arasındaki farkı hesaplama
    y_diff = lr_model.predict(X_test) - y_test

    # En yakın fiyat farkını bulma
    min_diff = np.abs(y_diff).min()

    # En yakın fiyat farkına sahip modeli bulma
    closest_model_idx = np.where(np.abs(y_diff) == min_diff)[0][0]
    closest_model = X_test.iloc[[closest_model_idx]]

    # Sonuçları yazdırma
    print("En yakın model:")
    print(closest_model)
    print("Tahmini fiyatı: ${:.2f}".format(lr_model.predict(closest_model)[0]))

def Brand_Lojistik_Regresyon(veriler):
    #----------------markalara lojistik regresyon---------

    # Marka sütununu kategorik sütuna dönüştürün
    veriler['Brand'] = pd.Categorical(veriler['Brand'])
    veriler['Brand'] = veriler['Brand'].cat.codes

    # Samsung markasına ait verileri seçin
    samsung_veriler = veriler[veriler['Brand'] == 11]

    # Özellikleri ve hedef değişkeni ayırma
    X = samsung_veriler.drop('Model', axis=1)
    y = samsung_veriler['Model']

    # Eğitim ve test setlerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lojistik regresyon modelini oluşturma ve eğitim
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Test setinde doğruluğu hesaplama
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Doğruluğu yazdırma
    print("Samsung için doğruluk:", accuracy)

    # Apple markasına ait verileri seçme
    apple_data = veriler[veriler['Brand'] == 0]

    # Özellikleri ve hedef değişkeni ayırma
    X_apple = apple_data.drop('Model', axis=1)
    y_apple = apple_data['Model']

    # Eğitim ve test setlerini ayırma
    X_train_apple, X_test_apple, y_train_apple, y_test_apple = train_test_split(X_apple, y_apple, test_size=0.2, random_state=42)

    # Lojistik regresyon modelini oluşturma ve eğitim
    lr_model_apple = LogisticRegression()
    lr_model_apple.fit(X_train_apple, y_train_apple)

    # Test verileri ile tahmin yapma ve doğruluk hesaplama
    y_pred_apple = lr_model_apple.predict(X_test_apple)
    accuracy_apple = accuracy_score(y_test_apple, y_pred_apple)

    # Sonuçları yazdırma
    print("Apple markası için doğruluk:", accuracy_apple)

    # Xiaomi markasına ait verileri seçme
    xiaomi_data = veriler[veriler['Brand'] == 15]

    # Özellikleri ve hedef değişkeni ayırma
    X_xiaomi = xiaomi_data.drop('Model', axis=1)
    y_xiaomi = xiaomi_data['Model']

    # Eğitim ve test setlerini ayırma
    X_train_xiaomi, X_test_xiaomi, y_train_xiaomi, y_test_xiaomi = train_test_split(X_xiaomi, y_xiaomi, test_size=0.2, random_state=42)

    # Lojistik regresyon modelini oluşturma ve eğitim
    lr_model_xiaomi = LogisticRegression()
    lr_model_xiaomi.fit(X_train_xiaomi, y_train_xiaomi)

    # Test verileri ile tahmin yapma ve doğruluk hesaplama
    y_pred_xiaomi = lr_model_xiaomi.predict(X_test_xiaomi)
    accuracy_xiaomi = accuracy_score(y_test_xiaomi, y_pred_xiaomi)

    # Sonuçları yazdırma
    print("Xiaomi markası için doğruluk:", accuracy_xiaomi)


def PCA_data(veriler):

    #----------------PCA--------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt


    # Yalnızca sayısal sütunları seçin
    numeric_cols = ['Storage ', 'RAM ', 'Battery Capacity (mAh)', 'Price']
    X = veriler[numeric_cols]

    # Verileri ölçeklendirin
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA modelini oluşturun
    pca = PCA(n_components=1)

    # PCA modelini eğitin
    pca.fit(X_scaled)

    # PCA dönüşümünü uygulayın
    X_pca = pca.transform(X_scaled)

    # Elde edilen PCA bileşenini veri kümesine ekleyin
    veriler['PCA Component'] = X_pca

    # PCA bileşeninin fiyatla ilişkisine bakın
    veriler.plot.scatter(x='PCA Component', y='Price')
    plt.show()
    Linear_Regression(veriler)
    Brand_Lojistik_Regresyon(veriler)

def main():
    print(veriler_read())


main()
