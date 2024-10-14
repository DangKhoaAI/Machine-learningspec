def decorator1(func):
    def wrapper(*args, **kwargs):
        print("Before calling the decorated function.")
        result = func(*args, **kwargs)
        print("After calling the decorated function.")
        return result
    return wrapper

def decorator2(func):
    @decorator1  # Áp dụng decorator1 bên trong decorator2
    def wrapper(*args, **kwargs):
        print("Inside decorator2 before calling the function.")
        result = func(*args, **kwargs)
        print("Inside decorator2 after calling the function.")
        return result
    return wrapper

@decorator1
def func1():
    print("This is func1.")

@decorator2
def func2():
    print("This is func2.")

# Gọi các hàm để xem kết quả
func1()
print()  # Chèn một dòng trống để dễ phân biệt
func2()
