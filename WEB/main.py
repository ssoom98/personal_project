from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from keras.layers import LeakyReLU
import joblib
import pandas as pd


# FastAPI 앱 생성
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 데이터 로드
food_data = pd.read_csv('../data/전처리후/reci_data_float.csv')
recipe_data = pd.read_csv('../data/전처리전/recipe_list.csv')

select_1 = food_data.loc[food_data['레시피구분'].isin(['밥류', '면류', '죽류'])]
select_2 = food_data.loc[food_data['레시피구분'].isin(['볶음류', '구이류', '조림류','찜류','부침류','튀김류'])]
select_3 = food_data.loc[food_data['레시피구분'].isin(['무침류', '김치류'])]
select_4 = food_data.loc[food_data['레시피구분'].isin(['국류'])]
select_5 = food_data.loc[food_data['레시피구분'].isin(['보조식'])]

# 예측할 모델
model = load_model("models/model.h5",custom_objects={
        'LeakyReLU': LeakyReLU,
        'mse': MeanSquaredError()
    })
loaded_objects = joblib.load("models/scaler_and_data.pkl")
scaler_X = loaded_objects["scaler_X"]

@app.get("/", response_class=HTMLResponse)
@app.get("/recommend_diet/", response_class=HTMLResponse)
async def form_handler(request:Request):
    """추천 식단 로직"""
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/recommend_diet/", response_class=HTMLResponse)
async def recommend_diet(
        request: Request,
        name: str= Form(...),
        age: int = Form(...),
        gender: str = Form(...),
        height: float = Form(...),
        weight: float = Form(...),
        activity_factor: float = Form(...)
):

    # 입력값 검증
    if not all([name,age, gender, height, weight, activity_factor]):
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "모든 필드를 올바르게 입력해주세요."
        })

    # BMI 계산
    BMI = round(weight / ((height * 0.01) ** 2),2)

    # BMR 계산(나이 키 몸무게 활동량에 정해진 계산식에 해당하는 권장섭취량 #비만인에게는 과도하게 높게나옴)
    if gender == '남자':
        BMR = round(662 - 9.53 * age + (15.91 * weight + 5.396 * height) * activity_factor,2)
    else:
        BMR = round(354 - 6.91 * age + (9.36 * weight + 7.26 * height) * activity_factor,2)

    # 권장 칼로리 계산
    if gender == '남자':
        if BMI < 13:
            recommended_calories = round((BMR + 500) * 0.33, 2)
            BMI_comment= "저체중입니다. 건강을 위해서 조금 더 드셔야겠어요."
        elif 13 <= BMI < 25:
            recommended_calories = round(BMR * 0.33,2)
            BMI_comment = "정상체중입니다."
        elif 25 <= BMI < 30:
            recommended_calories = round((BMR -300) * 0.33,2)
            BMI_comment = '과체중입니다. 조절할 필요가 있습니다.'
        else:
            recommended_calories = round((BMR - 500) * 0.33,2)
            BMI_comment = '비만입니다. 체중 조절을 위해 적게 드셔야해요.'
    else:
        if BMI < 22:
            recommended_calories = round((BMR + 500) * 0.33, 2)
            BMI_comment = '저체중입니다. 건강을 위해서 조금 더 드셔야겠어요.'
        elif 22 <= BMI < 34:
            recommended_calories = round(BMR * 0.33,2)
            BMI_comment = '정상체중입니다.'
        elif 34 <= BMI < 40:
            recommended_calories = round((BMR -300) * 0.33,2)
            BMI_comment = '과체중입니다. 조절할 필요가 있습니다.'
        else:
            recommended_calories = round((BMR - 500) * 0.33,2)
            BMI_comment = '비만입니다. 체중 조절을 위해 적게 드셔야해요.'

    remaining_calories = recommended_calories
    recommendations = []

    # 1. select_1 추천
    food = select_1.sample(1)
    food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 2. select_3 추천
    food = select_3.sample(1)
    food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 3. select_2에서 1개 추천
    food = select_2.sample(1)
    food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 4. select_4 추천
    food = select_4.sample(1)
    food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
    recommendations.append(food)
    remaining_calories -= food['열량'].sum()

    # 5. 열량이 부족하면 select_2에서 한번 더 추천
    if remaining_calories > 0:
        food = select_2[select_2['열량'] >= 150].sample(1)
        food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
        if food['열량'].sum() <= remaining_calories:
            recommendations.append(food)
            remaining_calories -= food['열량'].sum()

    # 5. 열량이 부족하면 select_5에서 추가
    if remaining_calories > 0:
        food = select_5.sample(1)
        food['열량'] = food['열량'].astype(float)  # 열량 값을 숫자로 변환
        if food['열량'].sum() <= remaining_calories:
            recommendations.append(food)
            remaining_calories -= food['열량'].sum()

    # 최종 추천 음식 합치기
    final_recommendations = pd.concat(recommendations)
    total_recommended_calories = round(final_recommendations['열량'].sum(), 2)

    # 결과 데이터프레임 생성 (총합 제거)
    result_df = final_recommendations[['레시피명', '열량']].copy()

    # 추천 레시피 세부 식재료 그룹화 및 정리
    grouped_recipes = recipe_data[recipe_data['레시피명'].isin(final_recommendations['레시피명'])]
    grouped_recipes = grouped_recipes[['레시피명', '식재료명', '식재료양']].set_index('레시피명').reset_index()

    # 일일 칼로리 잉여/적자
    calorie_balance = (recommended_calories*3)-BMR
    print(calorie_balance)
    # 독립변수 준비
    input_data = pd.DataFrame([{
        "age": age,
        "gender": 1 if gender == "남자" else 2,
        "height": height,
        "weight": weight,
        "activity_factor": activity_factor,
        "BMR": BMR,
        "energy_intake": recommended_calories*3,
        "calorie_balance": calorie_balance
    }])

    # 열 이름 통일 (scaler_X 학습 시 열 이름과 동일하게 설정)
    input_data.columns = scaler_X.feature_names_in_

    # 데이터 스케일링
    input_data_scaled = scaler_X.transform(input_data)

    # 모델 예측
    prediction_scaled = model.predict(input_data_scaled)

    # 예측 결과
    weight_change_prediction = round(prediction_scaled[0][0], 2)
    # 템플릿에 전달
    return templates.TemplateResponse("result.html", {
        "request": request,
        "BMR":BMR,
        "recommended_calories":recommended_calories,
        "name": name,
        "BMI": BMI,
        "recommended_diet": result_df.to_dict(orient="records"),
        "total_calories": total_recommended_calories,
        "grouped_recipes": grouped_recipes.to_dict(orient="records"),
        "weight_change_prediction": weight_change_prediction,
        "BMI_comment":BMI_comment
    })