package com.xiaoshuyui.kaidian.handler;


import com.xiaoshuyui.kaidian.entity.BaseEntity;
import com.xiaoshuyui.kaidian.entity.ExceptionEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;

@ControllerAdvice
public class GlobalExceptionHandler {

    private Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);


    @ExceptionHandler(value = Exception.class)
    @ResponseBody
    public ExceptionEntity defaultErrorHandler(HttpServletRequest req, Exception e) throws Exception {

        logger.error("", e);
        ExceptionEntity r = new ExceptionEntity();
        r.setMessage(e.getMessage());
        if (e instanceof org.springframework.web.servlet.NoHandlerFoundException) {
            r.setCode(404);
        } else {
            r.setCode(500);
        }
        r.setData(null);
        return r;
    }
}
