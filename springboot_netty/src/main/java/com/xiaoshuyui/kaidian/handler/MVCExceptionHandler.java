package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import com.xiaoshuyui.kaidian.exception.BussinessException;
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
import java.sql.SQLException;

@Slf4j
@RestController
@Order(9)
@ControllerAdvice
public class MVCExceptionHandler {

    private Logger logger = LoggerFactory.getLogger(MVCExceptionHandler.class);

    @ExceptionHandler(NoHandlerFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ExceptionEntity notFoundException(NoHandlerFoundException e) {

        logger.info("请求地址不存在: {}", e.getMessage());
        return new ExceptionEntity(StatusCode.NOT_FOUND, "not foundxxxxxxx", null);
    }

    @ExceptionHandler(ServletException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)

    public ExceptionEntity servletException(ServletException exception) {

        logger.info("请求方式或参数不合法: {}", exception.getMessage());
        return new ExceptionEntity(StatusCode.INVALID_REQUEST, "参数不合法", null);
    }
}
