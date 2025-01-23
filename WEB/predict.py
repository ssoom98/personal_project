from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# FastAPI 앱 생성
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 데이터 로드
food_data = pd.read_csv('../data/reci_data_float.csv')
recipe_data = pd.read_csv('../data/recipe_list.csv')

select_1 = food_data.loc[food_data['레시피구분'].isin(['밥류', '면류', '죽류'])]
select_2 = food_data.loc[food_data['레시피구분'].isin(['볶음류', '구이류', '조림류', '찜류', '부침류', '튀김류'])]
select_3 = food_data.loc[food_data['레시피구분'].isin(['무침류', '김치류'])]
select_4 = food_data.loc[food_data['레시피구분'].isin(['국류'])]
select_5 = food_data.loc[food_data['레시피구분'].isin(['보조식'])]

# 모델 및 스케일러 로드
model = load_model("../python/saved_model/pred_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")


@app.get("/", response_class=HTMLResponse)
@app.get("/recommend_diet/", response_class=HTMLResponse)
async def form_handler(request: Request):
    """입력 폼 표시"""
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/recommend_diet/", response_class=HTMLResponse)
async def recommend_diet(
        request: Request,
        name: str = Form(...),
        age: int = Form(...),
        gender: str = Form(...),
        height: float = Form(...),
        weight: float = Form(...),
        activity_factor: float = Form(...),
        energy_intake: float = Form(...),  # 에너지 섭취량
        calorie_balance: float = Form(...)  # 일일 칼로리 잉여/적자
):
    """추천 식단 및 체중 변화 예측"""
    # BMI 계산
    BMI = round(weight / ((height * 0.01) ** 2), 2)

    # BMR 계산
    if gender == '남자':
        BMR = round(662 - 9.53 * age + (15.91 * weight + 5.396 * height) * activity_factor, 2)
    else:
        BMR = round(354 - 6.91 * age + (9.36 * weight + 7.26 * height) * activity_factor, 2)

    # 권장 칼로리 계산
    if gender == '남자':
        if BMI < 13:
            recommended_calories = round((BMR + 500) * 0.33, 2)
        elif 13 <= BMI < 25:
            recommended_calories = round(BMR * 0.33, 2)
        elif 25 <= BMI < 30:
            recommended_calories = round(BMR * 0.8 * 0.33, 2)
        else:
            recommended_calories = round((BMR - 500) * 0.33, 2)
    else:
        if BMI < 22:
            recommended_calories = round((BMR + 500) * 0.33, 2)
        elif 22 <= BMI < 34:
            recommended_calories = round(BMR * 0.33, 2)
        elif 34 <= BMI < 40:
            recommended_calories = round(BMR * 0.8 * 0.33, 2)
        else:
            recommended_calories = round((BMR - 500) * 0.33, 2)

    # 독립변수 준비
    input_data = pd.DataFrame([{
        "age": age,
        "gender": 1 if gender == "남자" else 0,
        "BMR": BMR,
        "height": height,
        "weight": weight,
        "activity_factor": activity_factor,
        "energy_intake": energy_intake,
        "calorie_balance": calorie_balance
    }])

    # 데이터 스케일링
    input_data_scaled = scaler_X.transform(input_data)

    # 모델 예측
    prediction_scaled = model.predict(input_data_scaled)

    # 스케일 복원
    prediction_original = scaler_y.inverse_transform(prediction_scaled)

    # 예측 결과
    weight_change_prediction = round(prediction_original[0][0], 2)

    # 추천 식단 계산 (생략 가능)
    remaining_calories = recommended_calories
    recommendations = []

    # 1. select_1 추천
    food = select_1.sample(1)
    food['열량'] = food['열량'].astype(float)
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 2. select_3 추천
    food = select_3.sample(1)
    food['열량'] = food['열량'].astype(float)
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 3. select_2 추천
    food = select_2.sample(1)
    food['열량'] = food['열량'].astype(float)
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 최종 추천 음식 합치기
    final_recommendations = pd.concat(recommendations)
    total_recommended_calories = round(final_recommendations['열량'].sum(), 2)
    result_df = final_recommendations[['레시피명', '열량']].copy()

    # 추천 레시피 세부 식재료 그룹화 및 정리
    grouped_recipes = recipe_data[recipe_data['레시피명'].isin(final_recommendations['레시피명'])]
    grouped_recipes = grouped_recipes[['레시피명', '식재료명', '식재료양']].set_index('레시피명').reset_index()

    # 템플릿에 전달
    return templates.TemplateResponse("result.html", {
        "request": request,
        "name": name,
        "age": age,
        "gender": gender,
        "BMI": BMI,
        "BMR": BMR,
        "recommended_calories": recommended_calories,
        "energy_intake": energy_intake,
        "calorie_balance": calorie_balance,
        "weight_change_prediction": weight_change_prediction,
        "recommended_diet": result_df.to_dict(orient="records"),
        "total_calories": total_recommended_calories,
        "grouped_recipes": grouped_recipes.to_dict(orient="records")
    })
