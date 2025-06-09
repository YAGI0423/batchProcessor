import threading
from abc import *




class BatchProcessor:
    def __init__(self, batch_size: int, timeout: float) -> None:
        '''입력 데이터를 `{batch_size}` 크기의 배치 단위로 묶어,\n
        `self.process` 메소드로 처리하고 이를 다시 분해하여 입력처로 반환.\n
        * 주로 threading과 함께 사용
        * `process` 메소드 재정의 필요


        Args:
            batch_size (int): 배치 크기
            timeout (float): `timeout` 초과 시, 현재 등록된 데이터를 강제로 처리하여 반환
        '''
        self.batch_size = batch_size
        self.timeout = timeout

        self.__requests = dict()    #{thread_code(str): input(dict), ...}
        self.__responses = dict()    #{thread_code(str): response(any), ...}

        self.__lock = threading.Lock()
        self.__event = threading.Event()


    @abstractmethod
    def process(self, batch_input: tuple) -> tuple|list:
        '''Class 상속 시 사용자가 작성 해야하는 추상 메소드\n
        * return 값은 반드시 `batch_input`과 크기가 같아야 함
        * return 값의 데이터 형식은 반드시 tuple 또는 list 여야함


        Args:
            batch_input (tuple): >>> ({'input_0': 1}, 'input_1': 3, 'input_2': 8, ...)

        Returns:
            (tuple|list): 
        '''
        sum_ = 0
        for data in batch_input:
            sum_ += data['idx']
        return tuple(sum_ for _ in range(len(batch_input)))


    def __generate_batch(self, batch_size: int) -> dict:
        '''`self.__requests`로부터 `batch_size` 만큼 데이터를 추출하여 반환

        Args:
            batch_size (int): 추출할 데이터 수

        Returns:
            (dict): {thread_code(str): input(dict), ...}
        '''
        batch = dict()
        codes = tuple(self.__requests.keys())
        for code in codes[:batch_size]:
            batch[code] = self.__requests.pop(code)
        return batch


    def __regist_response(self, response_data: dict) -> None:
        '''`self.process`를 거친 응답(response)를 `self.__response`에 등록

        Args:
            response_data (dict): 응답 데이터. {thread_code(str): any, ...}
        '''
        for code, data in response_data.items():
            self.__responses[code] = data


    def __transponder(self, batch: dict) -> None:
        '''`응답기`로 `repeater`로부터 입력받은 batch 데이터를 `process`로 처리한 결과를 `response`에 등록\n
        * 등록 후, wait 상태의 thread 활성화

        Args:
            batch (dict): {thread_code(str): input(dict), ...}
        '''
        b_keys = tuple(batch.keys())
        b_values = tuple(batch.values())
        res = self.process(b_values)

        #check `self.process` return value
        assert type(res) in (tuple, list), 'The return value of the `process` method must be of type `tuple` or `list`.'
        assert len(res) == len(b_values), 'The length of the return value of the `process` method must be the same as the length of the input data.'

        #combine `key` with `value`
        res = {k: v for k, v in zip(b_keys, res)}
        self.__regist_response(res) #regist response
        self.__event.set()    #wakeup all threads
        

    def __repeater(self, input: dict, thread_code: str) -> any:
        '''`input`을 `self.requests`에 등록하고\n
        응답(response)을 반환

        Args:
            input (dict): 
            thread_code (str): thread를 구분하는 code

        Returns:
            (any|None): output
        '''
        self.__requests[thread_code] = input
        batch = dict()
        with self.__lock:
            if len(self.__requests) >= self.batch_size:
                batch = self.__generate_batch(self.batch_size)

        if batch:   #batch is not empty
            self.__transponder(batch)
            

        while thread_code not in self.__responses:
            if self.__event.wait(self.timeout):  #timeout
                self.__event.clear()
            else:
                with self.__lock:
                    batch = self.__generate_batch(len(self.__requests))

                if batch:
                    self.__transponder(batch)
        return self.__responses.pop(thread_code)
            

    def request(self, **input: any) -> any:
        '''입력 데이터(input)를 전송하고 Batch 처리된 데이터를 반환

        Args:
            input (any): 입력 데이터

        Returns:
            any: 전처리한 데이터 반환
        '''
        return self.__repeater(
            input=input,
            thread_code=hex(id(input))
        )



if __name__ == '__main__':
    from threading import Thread


    class TestBP(BatchProcessor):
        def process(self, batch_input: tuple) -> tuple|list:
            '''사용자 용도에 맞춰 작성 요함\n
            * return 값은 반드시 `{batch_input}`의 길이와 동일해야 함
            * return 값의 형태는 `tuple` 또는 `list` 여야함


            Attrs:
                batch_input(tuple): ({'input_0': 0}, {'input_1': 5}, ...)
            '''
            #예시로 input에 10을 곱한 값을 반환
            return [10. * item['input'] for item in batch_input]

        
    #timeout 초과 시, batch size에 도달하지 않았더라도,
    # 요청(request)된 데이터를 배치로 묶어 처리
    batch_processor = TestBP(
        batch_size=2,
        timeout=0.1,    
    )
    
    
    def worker(input: int) -> None:
        response = batch_processor.request(input=input)
        print(f'input: {input},\tresponse: {response}')

    
    threads = tuple(Thread(target=worker, kwargs={'input': i}) for i in range(5))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


'''
>>> input: 1,       response: 10.0
>>> input: 0,       response: 0.0
>>> input: 3,       response: 30.0
>>> input: 2,       response: 20.0
>>> input: 4,       response: 40.0
'''