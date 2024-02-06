import socket
from multiprocessing import shared_memory
import pickle  # 添加这一行
import time
def handle_client(client_socket, gpu_id):
    while True:
        request = client_socket.recv(1024).decode('utf-8').strip()
        if request.startswith("GPU_ID:"):
            gpu_id = request.split(":")[1]
        elif request == 'Training':
            print("Client is training...")
        elif request == 'Snapshotting':
            print("Client wants to snapshot...")
            try:
                time.sleep(10)
                sm = shared_memory.SharedMemory(name=str(gpu_id))
                buffer = sm.buf
                byte_data = buffer[:].tobytes()
                model_params = pickle.loads(byte_data)
                print(f"Received model params from GPU {gpu_id}")
            except FileNotFoundError:
                print(f"No model params received from GPU {gpu_id}")
                print(f"Shared Memory name attempted: {str(gpu_id)}")  
            except Exception as e:
                print(f"An error occurred: {e}")
        elif request == 'Exiting':
            print("Client is exiting...")
            break
    client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((socket.gethostbyname(socket.gethostname()), 4445))
    server.listen(5)
    print(f"[*] Listening on {socket.gethostbyname(socket.gethostname())}:4445")
    
    while True:
        client, addr = server.accept()
        print(f"[*] Accepted connection from: {addr[0]}:{addr[1]}")
        
        gpu_id = 0
        handle_client(client, gpu_id)

if __name__ == '__main__':
    main()
