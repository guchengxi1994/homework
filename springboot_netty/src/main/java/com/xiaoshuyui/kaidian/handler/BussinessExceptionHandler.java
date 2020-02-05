package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import com.xiaoshuyui.kaidian.exception.BussinessException;
import com.xiaoshuyui.kaidian.utils.StatusCode;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import java.sql.SQLException;

@Slf4j
@RestController
@Order(1)
@ControllerAdvice
public class BussinessExceptionHandler {

    private Logger logger = LoggerFactory.getLogger(BussinessExceptionHandler.class);

    @ExceptionHandler(BussinessException.class)
    @ResponseStatus(HttpStatus.OK)
    public ExceptionEntity defaultException(BussinessException e) {
        logger.error("业务异常: {}", e);
        return new ExceptionEntity(e.getExceptionCode(), e.getMessage(), null);
    }

    @ExceptionHandler(SQLException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ExceptionEntity defaultException(SQLException e) {
        logger.error("数据异常: {}", e);
        return new ExceptionEntity(StatusCode.DATABASE_ERROR, e.getMessage(), null);
    }


//    @ExceptionHandler(value = Exception.class)
//    @ResponseBody
//    public ExceptionEntity defaultErrorHandler(HttpServletRequest req, Exception e) throws Exception {
//
//        logger.error("", e);
//        ExceptionEntity r = new ExceptionEntity();
//        r.setMessage(e.getMessage());
//        if (e instanceof org.springframework.web.servlet.NoHandlerFoundException) {
//            r.setCode(404);
//        } else {
//            r.setCode(500);
//        }
//        r.setData(null);
//        return r;
//    }
}
