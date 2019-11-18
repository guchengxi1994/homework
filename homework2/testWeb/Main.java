package com.xiaoshuyui.testWeb;

import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;

public class Main {

    public static void main(String[] args) throws IOException {
        TestHandler th = new TestHandler();
        HttpServer server = HttpServer.create(new InetSocketAddress(8001),0);
        server.createContext("/test",th);
        server.start();



    }
}
