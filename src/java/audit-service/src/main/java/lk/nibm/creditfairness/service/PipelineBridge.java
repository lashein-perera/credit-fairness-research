package lk.nibm.creditfairness.service;

import org.springframework.stereotype.Service;

import java.util.Map;

/**
 * Bridge to the Python research pipeline.
 *
 * Simplest reliable approach: invoke the pipeline as a subprocess and exchange
 * JSON. Do not build a message queue for this.
 */
@Service
public class PipelineBridge {

    public Map<String, Object> invoke(String module, Map<String, Object> payload) {
        throw new UnsupportedOperationException("Week 6");
    }
}
