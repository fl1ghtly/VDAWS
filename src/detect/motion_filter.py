import cv2
from cv2 import typing

def filter_motion(prevFrame: typing.MatLike, nextFrame: typing.MatLike, threshold: int) -> typing.MatLike:
    """Returns a mask that contains the motion difference between two frames"""
    prev = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
    motion_mask = next - prev
    _, motion_mask = cv2.threshold(motion_mask, threshold, 255, cv2.THRESH_BINARY)
    
    return motion_mask
    
def _test_filter(file: str):
    cap = cv2.VideoCapture(file)

    ret, prev = cap.read()
    while (ret):
        ret, next = cap.read()
        if not ret: break
        
        motion_mask = filter_motion(prev, next, 2)
        cv2.imshow("Motion", motion_mask)
        cv2.waitKey()
        prev = next
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    _test_filter("./videos/test2/cam_R.mkv")               

