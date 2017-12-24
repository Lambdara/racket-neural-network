#lang racket

(provide neural-network%)

(require math/matrix)

(define neural-network%
  (class object%

    ;; List of layer sizes
    (init-rest init-topology)

    (define topology (list->vector init-topology))
    (define topology-size (vector-length topology))

    ;; Weights between neurons of different layers
    (define current-weights
      ;; Generate a vector, containing a vector for each set of weights between
      ;; layers which contains a vector for each neuron in the input-layer which
      ;; contains a vector with the weights for the neurons in the
      ;; output-layer.
      (letrec
          ((go
            (lambda (top)
              (if (or (null? top) (null? (cdr top)))
                  (list)
                  (cons (build-vector (car top)
                                      (lambda (to-size)
                                        (build-vector (cadr top)
                                                      (lambda (x) (- (* 2 (random))
                                                                     1)))))
                        (go (cdr top)))))))
        (list->vector (go (vector->list topology)))))
    (define (get-weight from-layer from-neuron to-neuron)
      (vector-ref (vector-ref (vector-ref current-weights
                                          from-layer)
                              from-neuron)
                  to-neuron))
    (define (set-weight! from-layer from-neuron to-neuron value)
      (vector-set! (vector-ref (vector-ref current-weights from-layer) from-neuron)
                   to-neuron
                   value))

    ;; Values for last input
    (define current-neuron-values
      (vector-map (lambda (size)
                    (make-vector size 'nil))
                  topology))
    (define (get-value layer neuron)
      (vector-ref (vector-ref current-neuron-values layer) neuron))
    (define (set-value! layer neuron value)
      (vector-set! (vector-ref current-neuron-values layer) neuron value))

    ;; Errors for last backpropagation
    (define current-neuron-errors
      (list->vector
       (map (lambda (size)
              (make-vector size 'nil))
            (cdr (vector->list topology)))))
    (define (get-error layer neuron)
      (vector-ref (vector-ref current-neuron-errors (sub1 layer)) neuron))
    (define (set-error! layer neuron value)
      (vector-set! (vector-ref current-neuron-errors (sub1 layer)) neuron value))

    ;; Feed input through network
    (define/public (feedforward input)
      (define (sigmoid x)
        (/ 1 (+ 1 (exp (- x)))))
      ;; Set first layer of values to input vector
      (for-each (lambda (index)
                  (set-value! 0 index (vector-ref input index)))
                (range (vector-length input)))
      ;; Set values of every other layer to the sigmoid of the weighted sum of
      ;; values of the previous layer.
      (for-each
       (lambda (from-layer)
         (for-each
          (lambda (to-neuron)
            (set-value! (add1 from-layer)
                        to-neuron
                        (sigmoid
                         (apply +
                                (map
                                 (lambda (from-neuron)
                                   (* (get-value from-layer
                                                 from-neuron)
                                      (get-weight from-layer
                                                  from-neuron
                                                  to-neuron)))
                                 (range (vector-ref topology from-layer)))))))
          (range (vector-ref topology (add1 from-layer)))))
       (range (sub1 topology-size))))

    ;; Get output from network
    (define/public (get-output)
      (vector-ref current-neuron-values (sub1 topology-size)))

    ;; Backpropagate errors
    (define/public (backpropagate gain targets)
      ;; Set last layer of errors
      (for-each
       (lambda (index)
         (set-error! (sub1 topology-size)
                     index
                     (let ((target-value (vector-ref targets index))
                           (actual-value (get-value (sub1 topology-size) index)))
                       ;; Error term for last layer
                       (* actual-value
                          (- 1 actual-value)
                          (- target-value actual-value)))))
       (range (vector-ref topology (sub1 topology-size))))
      ;; Set other layers of errors
      (for-each
       (lambda (to-layer)
         (for-each
          (lambda (from-neuron)
            (set-error! (sub1 to-layer)
                        from-neuron
                        (let ((value (get-value (sub1 to-layer) from-neuron)))
                          ;; Error term for hidden layers
                          (* value
                             (- 1 value)
                             (apply +
                                    (map (lambda (to-neuron)
                                           (* (get-weight (sub1 to-layer)
                                                          from-neuron
                                                          to-neuron)
                                              (get-error to-layer to-neuron)))
                                         (range (vector-ref topology to-layer))))))))
          (range (vector-ref topology (sub1 to-layer)))))
       (reverse (range 2 topology-size)))
      ;; Update weights
      (for-each
       (lambda (to-layer)
         (for-each
          (lambda (from-neuron)
            (for-each
             (lambda (to-neuron)
               (set-weight! (sub1 to-layer)
                            from-neuron
                            to-neuron
                            ;; Updated weight value
                            (+ (get-weight (sub1 to-layer) from-neuron to-neuron)
                               (* gain
                                  (get-error to-layer to-neuron)
                                  (get-value (sub1 to-layer) from-neuron)))))
             (range (vector-ref topology to-layer))))
          (range (vector-ref topology (sub1 to-layer)))))
       (reverse (range 1 topology-size))))

    (super-new)))
