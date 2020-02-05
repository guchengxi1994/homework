package com.xiaoshuyui.kaidian;

import com.xiaoshuyui.kaidian.entity.NettyServer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;


import java.net.InetSocketAddress;

@SpringBootApplication
@ServletComponentScan
public class SpringBootApplicationMain {


    public static void main(String[] args) {



        SpringApplication.run(SpringBootApplicationMain.class,args);

//        NettyServer nettyServer = new NettyServer();
//        nettyServer.start(new InetSocketAddress("127.0.0.1", 8090));
    }



}
