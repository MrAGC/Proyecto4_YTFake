from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reco_serving import get_recommendation_service


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

app = FastAPI(title="YTFake")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def current_user_id(request: Request) -> int | None:
    cookie_value = request.cookies.get("ytfake_user_id")
    if not cookie_value:
        return None
    try:
        return int(cookie_value)
    except ValueError:
        return None


def current_guest_id(request: Request) -> str | None:
    cookie_value = request.cookies.get("ytfake_guest_id")
    if not cookie_value:
        return None
    return cookie_value.strip() or None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
def login_page(request: Request):
    service = get_recommendation_service()
    suggested_users = service.get_login_users()
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "suggested_users": suggested_users,
            "app_name": "YTFake",
        },
    )


@app.post("/login")
def login_submit(user_id: int = Form(...)) -> RedirectResponse:
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie("ytfake_user_id", str(user_id), httponly=False, samesite="lax")
    response.delete_cookie("ytfake_guest_id")
    return response


@app.post("/guest-login")
def guest_login_submit(request: Request) -> RedirectResponse:
    guest_id = current_guest_id(request) or uuid4().hex
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie("ytfake_guest_id", guest_id, httponly=False, samesite="lax", max_age=60 * 60 * 24 * 30)
    response.delete_cookie("ytfake_user_id")
    return response


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("ytfake_user_id")
    response.delete_cookie("ytfake_guest_id")
    return response


@app.get("/home")
def home_page(request: Request):
    user_id = current_user_id(request)
    guest_id = current_guest_id(request)
    if user_id is None and guest_id is None:
        return RedirectResponse(url="/login", status_code=302)

    service = get_recommendation_service()
    page = service.home_page(user_id) if user_id is not None else service.guest_home_page(guest_id)
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "request": request,
            **page,
        },
    )


@app.get("/watch/{video_id}")
def watch_page(video_id: int, request: Request):
    user_id = current_user_id(request)
    guest_id = current_guest_id(request)
    if user_id is None and guest_id is None:
        return RedirectResponse(url="/login", status_code=302)

    service = get_recommendation_service()
    if user_id is not None:
        service.register_view(user_id, video_id)
    else:
        service.register_guest_view(guest_id, video_id)
    try:
        page = service.watch_page(user_id, video_id) if user_id is not None else service.guest_watch_page(guest_id, video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request=request,
        name="watch.html",
        context={
            "request": request,
            **page,
        },
    )


@app.get("/search")
def search_page(request: Request):
    user_id = current_user_id(request)
    guest_id = current_guest_id(request)
    if user_id is None and guest_id is None:
        return RedirectResponse(url="/login", status_code=302)

    service = get_recommendation_service()
    query = request.query_params.get("q", "")
    page = service.search_page(user_id, query) if user_id is not None else service.guest_search_page(guest_id, query)
    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "request": request,
            **page,
        },
    )


@app.get("/profile")
def profile_page(request: Request):
    user_id = current_user_id(request)
    guest_id = current_guest_id(request)
    if user_id is None and guest_id is None:
        return RedirectResponse(url="/login", status_code=302)

    service = get_recommendation_service()
    page = service.profile_page(user_id) if user_id is not None else service.guest_profile_page(guest_id)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            **page,
        },
    )


@app.get("/studio")
def studio_page(request: Request):
    user_id = current_user_id(request)
    guest_id = current_guest_id(request)
    if user_id is None and guest_id is None:
        return RedirectResponse(url="/login", status_code=302)

    service = get_recommendation_service()
    page = service.studio_page(user_id) if user_id is not None else service.guest_studio_page(guest_id)
    return templates.TemplateResponse(
        request=request,
        name="studio.html",
        context={
            "request": request,
            **page,
        },
    )
