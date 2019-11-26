package com.xiaoshuyui.nettyClient;


import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class);
        NettyClient client = new NettyClient();
        client.start();
    }
}
