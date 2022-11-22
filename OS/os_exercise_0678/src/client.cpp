#include <iostream>
#include "zmq.hpp"
#include <string>
#include <sstream>
#include <exception>
#include <signal.h>
#include "server_func.h"

void rl_create(zmq::socket_t& parent_socket, zmq::socket_t& socket, 
            int& port, int& create_id, int& id, int& pid)
{
    if(pid == -1) 
    {
        send_message(parent_socket, "Error: Cannot fork");
        pid = 0;
    } 
    else if(pid == 0) 
    {
        create_node(create_id,port);
    } 
    else 
    {
        id = create_id;
        send_message(socket, "pid");
        send_message(parent_socket, recieve_message(socket));
    }
}

void create(std::istringstream& command_stream, zmq::socket_t& parent_socket,
            zmq::socket_t& left_socket, zmq::socket_t& right_socket,
            int& left_pid, int& left_id, int& right_pid, int& right_id, int& id,
            int& left_port, int& right_port, std::string& request_string)
{
    int create_id;
    command_stream >> create_id;
    if(create_id == id) 
    {
        std::string message_string = "Error: Already exists";
        send_message(parent_socket, message_string);
    } 
    else if(create_id < id) 
    {
        if(left_pid == 0) 
        {
            left_pid = fork();
            rl_create(parent_socket, left_socket, left_port, 
                   create_id, left_id, left_pid);
        } 
        else 
        {
            send_message(left_socket, request_string);
            send_message(parent_socket, recieve_message(left_socket));
        }
    } 
    else 
    {
        if(right_pid == 0) 
        {
            right_pid = fork();
            rl_create(parent_socket, right_socket, right_port, 
                   create_id, right_id, right_pid);
        } 
        else 
        {
            send_message(right_socket, request_string);
            send_message(parent_socket, recieve_message(right_socket));
        }
    }
}

void rl_remove(zmq::socket_t& parent_socket, zmq::socket_t& socket, 
               int& port, int& delete_id, int& id, int& pid, 
               std::string& request_string)
{
    if(id == 0) 
    {
        send_message(parent_socket, "Error: Not found");
    } 
    else if(id == delete_id) 
    {
        send_message(socket, "kill_children");
        recieve_message(socket);
        kill(pid,SIGTERM);
        kill(pid,SIGKILL);
        id = 0;
        pid = 0;
        send_message(parent_socket, "Ok");

    } 
    else 
    {
        send_message(socket, request_string);
        send_message(parent_socket, recieve_message(socket));
    }
}

void remove(std::istringstream& command_stream, zmq::socket_t& parent_socket,
            zmq::socket_t& left_socket, zmq::socket_t& right_socket,
            int& left_pid, int& left_id, int& right_pid, int& right_id, int& id,
            int& left_port, int& right_port, std::string& request_string)
{
    int delete_id;
    command_stream >> delete_id;
    if (delete_id < id) 
    {
        rl_remove(parent_socket, left_socket, left_port, 
                  delete_id, left_id, left_pid, request_string);
    } 
    else 
    {
        rl_remove(parent_socket, right_socket, right_port, 
                  delete_id, right_id, right_pid, request_string);
    }
}

void rl_exec(zmq::socket_t& parent_socket, zmq::socket_t& socket, 
             int& id, int& pid, std::string& request_string)
{
    if(pid == 0) 
    {
        std::string recieve_message = "Error:" + std::to_string(id);
        recieve_message += ": Not found";
        send_message(parent_socket, recieve_message);
    } 
    else 
    {
        send_message(socket, request_string);
        send_message(parent_socket, recieve_message(socket));
    }
}

void exec(std::istringstream& command_stream, zmq::socket_t& parent_socket,
          zmq::socket_t& left_socket, zmq::socket_t& right_socket,
          int& left_pid, int& left_id, int& right_pid, int& right_id, int& id,
          int& left_port, int& right_port, std::string& request_string)
{
    int exec_id;
    command_stream >> exec_id;
    if(exec_id == id) 
    {
        int n;
        command_stream >> n;
        int sum = 0;
        for(int i = 0; i < n; ++i) 
        {
            int cur_num;
            command_stream >> cur_num;
            sum += cur_num;
        }
        std::string recieve_message = "Ok:" + std::to_string(id) + ":";
        recieve_message += std::to_string(sum);
        send_message(parent_socket, recieve_message);
        
    } 
    else if(exec_id < id) 
    {
        rl_exec(parent_socket, left_socket, exec_id,
                left_pid, request_string);
    } 
    else 
    {
        rl_exec(parent_socket, right_socket, exec_id,
                right_pid, request_string);
    }
}

bool repl(std::string& str, std::string& from, std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
    {
        return false;
    }
    str.replace(start_pos, from.length(), to);
    return true;
}

void heartbit(std::istringstream& command_stream, zmq::socket_t& parent_socket, 
              zmq::socket_t& left_socket, zmq::socket_t& right_socket, int& left_pid, 
              int& right_pid, int& left_id, int& right_id, 
              int& id, std::string& request_string)
{
    bool flag = true;
    int time = 0;
    int my_id = 0;
    command_stream >> time >> my_id;
    left_socket.setsockopt(ZMQ_RCVTIMEO, 4*time);
    right_socket.setsockopt(ZMQ_RCVTIMEO, 4*time);
    if(my_id == id)
    {
        send_message(parent_socket, std::to_string(id));
    }
    else if(my_id < id)
    {
        send_message(left_socket, request_string);
        send_message(parent_socket, recieve_message(left_socket));
    }
    else if(my_id > id)
    {
        send_message(right_socket, request_string);
        send_message(parent_socket, recieve_message(right_socket));
    }
}

void kill_children(zmq::socket_t& parent_socket, zmq::socket_t& left_socket,
                   zmq::socket_t& right_socket, int& left_pid, int& right_pid)
{
    if(left_pid == 0 && right_pid == 0) 
    {
        send_message(parent_socket, "Ok");
    } 
    else 
    {
        if(left_pid != 0) 
        {
            send_message(left_socket, "kill_children");
            recieve_message(left_socket);
            kill(left_pid,SIGTERM);
            kill(left_pid,SIGKILL);
        }
        if(right_pid != 0) 
        {
            send_message(right_socket, "kill_children");
            recieve_message(right_socket);
            kill(right_pid,SIGTERM);
            kill(right_pid,SIGKILL);
        }
        send_message(parent_socket, "Ok");
    }
}

int main(int argc, char** argv) 
{
    int id = std::stoi(argv[1]);
    int parent_port = std::stoi(argv[2]);
    zmq::context_t context(3);
    zmq::socket_t parent_socket(context, ZMQ_REP);
    parent_socket.connect(get_port_name(parent_port));

    int left_pid = 0;
    int right_pid = 0;
    int left_id = 0;
    int right_id = 0;
    int linger = 0;
    
    zmq::socket_t left_socket(context, ZMQ_REQ);
    zmq::socket_t right_socket(context, ZMQ_REQ);
    left_socket.setsockopt(ZMQ_SNDTIMEO, 2000);
    left_socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    right_socket.setsockopt(ZMQ_SNDTIMEO, 2000);
    right_socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

    int left_port = bind_socket(left_socket);
    int right_port = bind_socket(right_socket);

    while(true) 
    {
        std::string request_string = recieve_message(parent_socket);
        std::istringstream command_stream(request_string);
        std::string command;
        command_stream >> command;

        if(command == "id") 
        {
            std::string parent_string = "Ok:" + std::to_string(id);
            send_message(parent_socket, parent_string);
        } 
        else if(command == "pid") 
        {
            std::string parent_string = "Ok:" + std::to_string(getpid());
            send_message(parent_socket, parent_string);
        } 
        else if(command == "create") 
        {
            create(command_stream, parent_socket, left_socket, right_socket,
                   left_pid, left_id, right_pid, right_id, id, left_port,
                   right_port, request_string);
        } 
        else if(command == "remove") 
        {
            remove(command_stream, parent_socket, left_socket, right_socket,
                   left_pid, left_id, right_pid, right_id, id, left_port,
                   right_port, request_string);
        } 
        else if(command == "exec") 
        {
            exec(command_stream, parent_socket, left_socket, right_socket,
                 left_pid, left_id, right_pid, right_id, id, left_port,
                 right_port, request_string);
        }
        else if(command == "heartbit") 
        {
            heartbit(command_stream, parent_socket, left_socket, right_socket,
                     left_pid, right_pid, left_id, right_id, id, request_string);
        } 
        else if(command == "kill_children") 
        {
            kill_children(parent_socket, left_socket, 
                          right_socket,  left_pid, right_pid); 
        }
        if(parent_port == 0) { break; }
    }
}
