from fastapi import FastAPI
from routes import user_routes, model_response_routes, oauth_routes
from database import engine, Base

# Initialize DB tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(title="Discover Islam")

# Include Routers
app.include_router(oauth_routes.router)
app.include_router(user_routes.router)
app.include_router(model_response_routes.router)

