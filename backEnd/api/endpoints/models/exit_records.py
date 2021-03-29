from pydantic import BaseModel

class exit_Records(BaseModel):
    carplate_no: str
    Out_time: str