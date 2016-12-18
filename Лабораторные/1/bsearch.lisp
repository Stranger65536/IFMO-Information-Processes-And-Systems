(defun binary-search (value array &optional (low 0) (high (1- (length array))))
  (if (< high low)
      nil
      (let ((middle (floor (/ (+ low high) 2))))
 
        (cond ((> (aref array middle) value)
               (binary-search value array low (1- middle)))
 
              ((< (aref array middle) value)
               (binary-search value array (1+ middle) high))
 
              (t middle)))))

(setq array (make-array '(11) 
						:initial-contents 
						'(1 2 4 8 16 32 64 128 256 512 1024)))
(setq value 4)			
(setq result (binary-search value array))
(format t "Position of ~S~%in ~S~%is ~S~%" value array result)

(setq value 0)			
(setq result (binary-search value array))
(format t "Position of ~S~%in ~S~%is ~S~%" value array result)