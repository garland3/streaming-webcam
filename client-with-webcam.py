import select
import socket
import cv2
import pickle
import struct

# Initialize socket and camera
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8485))
server_socket.listen(1)
server_socket.setblocking(False)  # Set socket to non-blocking

cap = cv2.VideoCapture(0)

exit_flag = False

while not exit_flag:
    print("Waiting for a connection...")
    ready_to_read, _, _ = select.select([server_socket], [], [], 5)
    
    if ready_to_read:
        conn, addr = ready_to_read[0].accept()
        print(f"Connected to {addr}")

        try:
            while True:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (640, 480))

                # Serialize frame
                data = pickle.dumps(frame)
                data_length = len(data)

                # Send length of data and data with delimiter
                conn.sendall(struct.pack("<L", data_length) + data + b'ENDFRAME')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    exit_flag = True
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("Closing connection.")
            conn.close()
            
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        exit_flag = True
        break
# Clean up
cap.release()
server_socket.close()
