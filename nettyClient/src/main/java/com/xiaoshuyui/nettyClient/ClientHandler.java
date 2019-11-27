package com.xiaoshuyui.nettyClient;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import lombok.extern.slf4j.Slf4j;

import java.nio.charset.Charset;

@Slf4j
public class ClientHandler extends ChannelInboundHandlerAdapter {
    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
//        log.info("客户端Active .....");
        System.out.println("客户端Active .....");
        int i = 0;

        while (i<200){
            ByteBuf buf = getByteBuf(ctx);
            ctx.channel().writeAndFlush(buf);
            i++;
        }

    }

    private ByteBuf getByteBuf(ChannelHandlerContext ctx){
        byte[] bytes = "abcdefg\n".getBytes(Charset.forName("utf-8"));
        ByteBuf buf = ctx.alloc().buffer();
        buf.writeBytes(bytes);
        return buf;
    }

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
//        log.info("客户端收到消息: {}", msg.toString());
        System.out.println("客户端收到消息: {}"+ msg.toString());
    }


    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
        cause.printStackTrace();
        ctx.close();
    }

}
