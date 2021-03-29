from pydantic import BaseModel

class Users(BaseModel):
    carplate_no: str
    name: str
    emp_id: str