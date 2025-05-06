# Path: app.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from Electric_Vehicle_Prediction.constants import APP_HOST, APP_PORT
from Electric_Vehicle_Prediction.pipeline.prediction_pipeline import ElectricVehicleData, EVPredictor
from Electric_Vehicle_Prediction.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.model_year: Optional[int] = None
        self.make: Optional[str] = None
        self.model: Optional[str] = None
        self.electric_vehicle_type: Optional[str] = None
        self.clean_alternative_fuel_vehicle_eligibility: Optional[str] = None
        self.base_msrp: Optional[float] = None

    async def get_ev_data(self):
        form = await self.request.form()
        self.model_year = form.get("model_year")
        self.make = form.get("make")
        self.model = form.get("model")
        self.electric_vehicle_type = form.get("electric_vehicle_type")
        self.clean_alternative_fuel_vehicle_eligibility = form.get("clean_alternative_fuel_vehicle_eligibility")
        self.base_msrp = form.get("base_msrp")


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
        "EV_Template.html", {"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_ev_data()

        ev_input_data = ElectricVehicleData(
            model_year=form.model_year,
            make=form.make,
            model=form.model,
            electric_vehicle_type=form.electric_vehicle_type,
            clean_alternative_fuel_vehicle_eligibility=form.clean_alternative_fuel_vehicle_eligibility,
            base_msrp=form.base_msrp,
        )

        ev_df = ev_input_data.get_ev_input_data_frame()

        model_predictor = EVPredictor()

        electric_range_pred = model_predictor.predict(dataframe=ev_df)

        return templates.TemplateResponse(
            "EV_Template.html",
            {"request": request, "Electric Range": electric_range_pred},
        )

    except Exception as e:
        return {"error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)