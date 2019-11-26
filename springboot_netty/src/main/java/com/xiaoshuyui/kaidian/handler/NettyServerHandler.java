package com.xiaoshuyui.kaidian.handler;

import com.alibaba.fastjson.JSON;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetSocketAddress;
import java.util.Map;


@Slf4j
public class NettyServerHandler extends ChannelInboundHandlerAdapter {

    private Logger log = LoggerFactory.getLogger(NettyServerHandler.class);;

    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
//        super.channelActive(ctx);
//        log.info("aaa");
        log.info("系统激活");
//        System.out.println("aaa");
    }


    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
//        super.channelRead(ctx, msg);
        log.info(msg.toString());
//        System.out.println(msg.toString());
        ctx.write("bbb");
        ctx.flush();
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
//        super.exceptionCaught(ctx, cause);
        InetSocketAddress insocket = (InetSocketAddress) ctx.channel().remoteAddress();
        String clientIp = insocket.getAddress().getHostAddress();
        log.info("客户端[ip:" + clientIp + "]连接出现异常，服务器主动关闭连接。。。");
        cause.printStackTrace();
        ctx.close();
    }


    protected void channelRead0(ChannelHandlerContext ctx,String msg){
        StringBuilder sb = null;
        Map<String,Object> result = null;

        try {
            sb = new StringBuilder();
            result = JSON.parseObject(msg);
            sb.append(result);
            sb.append("解析成功");
            sb.append("\n");
            ctx.writeAndFlush(sb);
        }catch (Exception e){
//            e.printStackTrace();
            String errorCode = "-1\n";
            ctx.writeAndFlush(errorCode);
            log.error("报文解析失败: " + e.getMessage());
        }


    }
}
