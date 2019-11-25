package com.xiaoshuyui.kaidian.handler;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import lombok.extern.slf4j.Slf4j;

//import java.util.logging.Logger;

@Slf4j
public class NettyServerHandler extends ChannelInboundHandlerAdapter {

//    private  Logger log;

    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
//        super.channelActive(ctx);
//        log.info("aaa");
        System.out.println("aaa");
    }


    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
//        super.channelRead(ctx, msg);
//        log.info(msg.toString());
        System.out.println(msg.toString());
        ctx.write("bbb");
        ctx.flush();
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
//        super.exceptionCaught(ctx, cause);
        cause.printStackTrace();
        ctx.close();
    }
}
