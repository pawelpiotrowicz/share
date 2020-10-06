void memcpy1(void *pvDest, void *pvSrc, size_t nBytes) {

      typedef long long __int64;

      /* We don't care about this scenario at this moment - WIP */  
      if(nBytes<sizeof(__int64_t))
	  {
		    /* this code can be faster - WIP */
			char* b = reinterpret_cast<char*>(pvSrc);
            char* e = b + nBytes;
			char* out = reinterpret_cast<char*>(pvDest); 
            std::copy(b,e,out); 
	        return; 
	  }	  

      /* Below is real gain */
      for(size_t i=0; i + sizeof(__int64) <= nBytes;i+=sizeof(__int64)) {
		_mm_stream_si64 ((reinterpret_cast<__int64*>(reinterpret_cast<char*>(pvDest)+i)), *(reinterpret_cast<__int64*>((reinterpret_cast<char*>(pvSrc) + i))));
	  } 

      /* We don't care about this scenario at this moment - WIP */  
      size_t left_bytes = nBytes % sizeof(__int64); 

      if(left_bytes)
	  {
		  /* this code can be faster - WIP */ 
	     char* e = reinterpret_cast<char*>(pvSrc) + nBytes;
         char* b = e - left_bytes;
	  	 char* out = reinterpret_cast<char*>(pvDest) + nBytes - left_bytes; 
         std::copy(b,e,out);  				  
	  }

}
