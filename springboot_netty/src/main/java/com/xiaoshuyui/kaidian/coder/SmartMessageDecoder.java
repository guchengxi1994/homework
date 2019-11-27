package com.xiaoshuyui.kaidian.coder;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;

import javax.xml.bind.DatatypeConverter;
import java.util.Arrays;
import java.util.List;

public class SmartMessageDecoder extends MessageToMessageDecoder<ByteBuf> {

    private byte[] remainingBytes;
    private static byte[] HEAD_DATA = new byte[]{0x75, 0x72};

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf byteBuf, List<Object> list) throws Exception {
        ByteBuf currBB = null;
        if (null == remainingBytes) {
            currBB = byteBuf;
        } else {
            byte[] tb = new byte[remainingBytes.length + byteBuf.readableBytes()];
            System.arraycopy(remainingBytes, 0, tb, 0, remainingBytes.length);
            byte[] vb = new byte[byteBuf.readableBytes()];
            byteBuf.readBytes(vb);
            System.arraycopy(vb, 0, tb, remainingBytes.length, vb.length);
            currBB = Unpooled.copiedBuffer(tb);
        }

        while (currBB.readableBytes() > 0) {
            if (!doDecode(ctx, currBB, list)) {
                break;
            }
        }

        if(currBB.readableBytes()>0){
            remainingBytes = new byte[currBB.readableBytes()];
            currBB.readBytes(remainingBytes);
        }else{
            remainingBytes = null;
        }
    }

    private boolean doDecode(ChannelHandlerContext ctx,ByteBuf msg,List<Object> out){
        if(msg.readableBytes()<2){
            return false;
        }
        msg.markReaderIndex();
        byte[] header = new byte[2];
        msg.readBytes(header);
        byte[] dataLength = new byte[2];
        msg.readBytes(dataLength);

        if(!Arrays.equals(header,HEAD_DATA)){
            return false;
        }

        int len = Integer.parseInt(DatatypeConverter.printHexBinary(dataLength),16);

        if(msg.readableBytes()<len-4){
            msg.resetReaderIndex();
            return  false;
        }

        msg.resetReaderIndex();
        byte[] body = new byte[len];
        msg.readBytes(body);
        out.add(Unpooled.copiedBuffer(body));
        if(msg.readableBytes()>0){
            return true;
        }
        return false;

    }
}
