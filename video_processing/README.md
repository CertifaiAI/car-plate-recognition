Video Processing for dash cams videos

Install required package
``` 
pip3 install -r requirements.txt 
```

1. Without models
```  
python3 extract_no_model.py --video dashcam.MP4 --interval 5 
``` 

Flags:
- video     = video path
- interval  = Time (in seconds) to capture frame

2. With models
''' '''
Flags:

3. Convert PNG to JPG or JPEG
```
python3 convert_png2jpg.py --image test.png 
```