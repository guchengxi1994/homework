package com.xiaoshuyui.kaidian.listener;


import com.xiaoshuyui.kaidian.entity.NettyServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;
import javax.servlet.annotation.WebListener;
import java.net.InetSocketAddress;

@WebListener
public class NettyServerListener implements ServletContextListener {

    private Logger log = LoggerFactory.getLogger(NettyServerListener.class);

    @Value("${netty.port}")
    private int port;

    @Value("${netty.address}")
    private String address;


    @Autowired
    private NettyServer nettyServer;

    @Override
    public void contextInitialized(ServletContextEvent sce) {
        log.info("ServletContex初始化...");

        Thread thread = new Thread(new NettyServerThread());

        thread.start();
    }

    @Override
    public void contextDestroyed(ServletContextEvent sce) {
        log.info("stopped");
    }

    private class NettyServerThread implements Runnable{


        @Override
        public void run() {
            InetSocketAddress ip = new InetSocketAddress(address,port);
            nettyServer.start(ip);
        }
    }
}
