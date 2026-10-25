"""Example tasks: typed outputs plus the prompt template each one uses."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from aiplay import prompts
from aiplay.providers.base import Provider
from aiplay.structured import Extraction, extract


class Experience(BaseModel):
    employer: str
    title: str
    start: str | None = Field(None, description="As written, e.g. 2021-03 or Mar 2021")
    end: str | None = Field(None, description="As written, or 'present'")


class Resume(BaseModel):
    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    skills: list[str] = Field(default_factory=list, description="Technical skills as written in the resume")
    total_years_experience: float | None = Field(None, description="Only if stated or derivable from listed dates")
    experience: list[Experience] = Field(default_factory=list)


class Category(StrEnum):
    ENGINEERING = "engineering"
    DATA = "data"
    DESIGN = "design"
    PRODUCT = "product"
    SALES = "sales"
    OPERATIONS = "operations"
    OTHER = "other"


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class JobClassification(BaseModel):
    category: Category
    seniority: Seniority
    remote: bool
    key_skills: list[str] = Field(description="Up to five skills the role requires")


async def extract_resume(provider: Provider, resume_text: str) -> Extraction[Resume]:
    template = prompts.load("resume_extract")
    return await extract(provider, Resume, template.render(resume_text=resume_text), system=template.system)


async def classify_job(provider: Provider, job_description: str) -> Extraction[JobClassification]:
    template = prompts.load("job_classify")
    return await extract(provider, JobClassification, template.render(job_description=job_description), system=template.system)
