package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import com.xiaoshuyui.kaidian.exception.BussinessException;
import com.xiaoshuyui.kaidian.utils.StatusCode;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.NoHandlerFoundException;

import javax.servlet.ServletException;
import java.sql.SQLException;

//@Configuration
public class ExceptionHandlerConfiguration {
    private static Logger logger = LoggerFactory.getLogger(ExceptionHandlerConfiguration.class);

//    @Slf4j
//    @RestController
//    @Order(1)
//    @ControllerAdvice

//    public static class BusinessExceptionHandler {
//        @ExceptionHandler(BussinessException.class)
//        @ResponseStatus(HttpStatus.OK)
//        public ExceptionEntity defaultException(BussinessException e) {
//            logger.error("业务异常: {}", e);
//            return new ExceptionEntity(e.getExceptionCode(), e.getMessage(), null);
//        }
//
//        @ExceptionHandler(SQLException.class)
//        @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
//        public ExceptionEntity defaultException(SQLException e) {
//            logger.error("数据异常: {}", e);
//            return new ExceptionEntity(StatusCode.DATABASE_ERROR, e.getMessage(), null);
//        }
//
//
//    }

//    @Slf4j
//    @RestController
//    @Order(9)
//    @ControllerAdvice

//    public static class MVCExceptionHandler {
//
////        @ExceptionHandler(NoHandlerFoundException.class)
////        @ResponseStatus(HttpStatus.NOT_FOUND)
////        public ExceptionEntity notFoundException(NoHandlerFoundException e) {
////
////            logger.info("请求地址不存在: {}", e.getMessage());
////            return new ExceptionEntity(StatusCode.NOT_FOUND, "not foundxxxxxxx", null);
////        }
////
////        @ExceptionHandler(ServletException.class)
////        @ResponseStatus(HttpStatus.BAD_REQUEST)
////
////        public ExceptionEntity servletException(ServletException exception) {
////
////            logger.info("请求方式或参数不合法: {}", exception.getMessage());
////            return new ExceptionEntity(StatusCode.INVALID_REQUEST,"参数不合法",null);
////        }
//
//    }


//    @Slf4j
//    @RestController
//    @Order(98)
//    @ControllerAdvice
//    public static class RuntimeExceptionHandler {
//
//        /**
//         * 缺省运行时异常
//         *
//         * @param exception
//         * @return
//         */
//        @ExceptionHandler(RuntimeException.class)
//        @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
//        public ExceptionEntity runtimeException(RuntimeException exception) {
//
//            logger.error("Spring Boot 未知错误", exception);
//            return new ExceptionEntity(StatusCode.ERROR,"未知错误",null);
//        }
//    }
}
