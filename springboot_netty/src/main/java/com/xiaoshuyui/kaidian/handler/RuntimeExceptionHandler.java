package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import com.xiaoshuyui.kaidian.utils.StatusCode;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.NoHandlerFoundException;

import javax.servlet.ServletException;

@Slf4j
@RestController
@Order(98)
@ControllerAdvice
public class RuntimeExceptionHandler {

    private Logger logger = LoggerFactory.getLogger(RuntimeExceptionHandler.class);

    @ExceptionHandler(RuntimeException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ExceptionEntity runtimeException(RuntimeException exception) {

        logger.error("Spring Boot 未知错误", exception);
        return new ExceptionEntity(StatusCode.ERROR,"未知错误",null);
    }
}
