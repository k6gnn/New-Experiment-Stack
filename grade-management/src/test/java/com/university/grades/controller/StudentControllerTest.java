package com.university.grades.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.university.grades.model.Student;
import com.university.grades.service.StudentService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Arrays;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(StudentController.class)
class StudentControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private StudentService studentService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void getAllStudents_shouldReturn200WithStudentList() throws Exception {
        Student s1 = new Student("Alice", 8.5);
        s1.setId(1L);
        Student s2 = new Student("Bob", 6.0);
        s2.setId(2L);

        when(studentService.getAllStudents()).thenReturn(Arrays.asList(s1, s2));

        mockMvc.perform(get("/students"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].name").value("Alice"))
                .andExpect(jsonPath("$[1].name").value("Bob"));
    }

    @Test
    void createStudent_shouldReturn201WithCreatedStudent() throws Exception {
        Student newStudent = new Student("Charlie", 9.0);
        newStudent.setId(3L);

        when(studentService.createStudent(any(Student.class))).thenReturn(newStudent);

        mockMvc.perform(post("/students")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(newStudent)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("Charlie"))
                .andExpect(jsonPath("$.grade").value(9.0));
    }

    @Test
    void getStudentGrade_shouldReturn200WithGradeWhenStudentExists() throws Exception {
        when(studentService.getGradeById(1L)).thenReturn(Optional.of(8.5));

        mockMvc.perform(get("/students/1/grade"))
                .andExpect(status().isOk())
                .andExpect(content().string("8.5"));
    }

    @Test
    void getStudentGrade_shouldReturn404WhenStudentNotFound() throws Exception {
        when(studentService.getGradeById(99L)).thenReturn(Optional.empty());

        mockMvc.perform(get("/students/99/grade"))
                .andExpect(status().isNotFound());
    }

    @Test
    void createStudent_shouldReturn400WhenNameIsBlank() throws Exception {
        Student invalidStudent = new Student("", 8.5);

        mockMvc.perform(post("/students")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(invalidStudent)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void createStudent_shouldReturn400WhenGradeExceedsMaximum() throws Exception {
        Student invalidStudent = new Student("Dave", 11.0);

        mockMvc.perform(post("/students")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(invalidStudent)))
                .andExpect(status().isBadRequest());
    }
}
