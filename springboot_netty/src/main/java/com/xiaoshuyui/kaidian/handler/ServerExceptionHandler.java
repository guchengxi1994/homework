package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import com.xiaoshuyui.kaidian.utils.StatusCode;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;


public class ServerExceptionHandler {

    private static Logger logger = LoggerFactory.getLogger(ServerExceptionHandler.class);
    @Slf4j
    @RestController
    @ControllerAdvice
    public static class RuntimeExceptionHandler{

        @ExceptionHandler(RuntimeException.class)
        @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)

        public ExceptionEntity runtimeException(RuntimeException r){
            logger.error("Spring Boot 未知错误", r);
            return new ExceptionEntity(StatusCode.ERROR,r.getMessage(),null);
        }
    }
}
