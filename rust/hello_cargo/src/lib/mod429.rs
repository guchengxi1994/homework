pub mod test{

    pub fn t1(){
        println!("hello 429");
    }

    // pub fn t2(a: i32) -> Result<(), Error> {
    //     let condition:bool = true;
    //     let mut num = if condition {5} else {a};
    //     Ok(())
    // }

    pub fn t2(){
        for i in (1..5).rev() {
            println!("{}", i);
        }
        println!("owee");
    }
}