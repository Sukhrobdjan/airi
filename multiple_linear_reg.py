import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_exam_score(hours, attendance):

    data = {
        "hours_study": [2, 3, 4, 5, 6, 7, 8],
        "attendance": [80, 82, 85, 90, 92, 95, 98],
        "exam_score": [50, 55, 60, 65, 70, 75, 80]  # Kutilayotgan natija
    }

    df = pd.DataFrame(data)  # ma'lumotlar to'plamini yaratish

    X = df[["hours_study", "attendance"]]  # mustaqil o'zgaruvchilar
    y = df['exam_score']  # bog'liq o'zgaruvchi

    model = LinearRegression()  # model yaratish
    model.fit(X, y)  # modelni o'qitish
    print("Koeffitsent(w1, w2):", model.coef_)
    print("Intercept:", model.intercept_)

    predict = model.predict([[hours, attendance]]) # bashorat qilish
    return predict[0]


# Misol uchun
print('Talaba 5 soat o\'qib, 90% qatnashsa, imtihon bahosi:', predict_exam_score(5, 90))
# print('Talaba 7 soat o\'qib, 95% qatnashsa, imtihon bahosi:', predict_exam_score(7, 95))


def apartament_price(size, rooms, location_score):

    data = {
        "size": [50, 60, 70, 80, 90, 100, 110],
        "rooms":[1, 2, 2, 3, 3, 4, 4],
        "location_score": [5, 6, 7, 8, 7, 9, 10],
        "price": [100, 120, 140, 160, 180, 200, 220]  # Kutilayotgan natija 
    }

    df = pd.DataFrame(data)
    
    x = df[["size", "rooms", "location_score"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(x,y)

    print("koeffitsentilar(w1, w2, w3):", model.coef_)
    print("Intercept:", model.intercept_)

    predicted_value = model.predict([[size, rooms, location_score]])
    return predicted_value[0]

print("Agar 80 kv.m, 3 xonali va joylashuvi 8 bo'lsa, narxi:", apartament_price(80, 3, 8))
print("Agar 100 kv.m, 4 xonali va joylashuvi 9 bo'lsa, narxi:", apartament_price(100, 4, 9))