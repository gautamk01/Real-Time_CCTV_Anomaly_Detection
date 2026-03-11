import threading
import time 

def check(name,n):
    for i in range(n):
        print(f"Thread {name} : ",i)
        time.sleep(1)

def main():
    threads = []
    for i in range(3):
        thread = threading.Thread(target=check,args=(i,3))
        threads.append(thread)
        thread.start()
    for i in threads:
        i.join()
    print("Main thread finished here")

if __name__ == "__main__":
    main()